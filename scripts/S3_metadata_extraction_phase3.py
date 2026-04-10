#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3 Phase 3: 元数据提取（增强主键版）

核心调整：
- 提高 year/title/school/major 的完整度
- 增加更稳定的 paper_hash / source_doc_id 导出
- 导出可直接供后续困难样本表使用的主键

输入: phase3/01_qc_output/, phase3/02_paragraphs/
输出: phase3/03_metadata/documents_metadata.json
"""

import os
import re
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict


@dataclass
class DocumentMetadata:
    """文档元数据结构（Phase 3 增强版）"""
    # 主键
    doc_id: str                    # 文档ID（文件名）
    paper_hash: str                # 内容哈希（稳定主键）
    source_doc_id: str             # 源文档标识
    
    # 基本信息
    year: int
    title: str
    school: str
    major: str
    author: str
    
    # 内容统计
    char_count: int
    para_count: int
    chapter_count: int
    
    # 路径信息
    source_path: str
    relative_path: str
    
    # 质量标记
    has_complete_metadata: bool    # 元数据是否完整
    missing_fields: List[str]      # 缺失的字段


class MetadataExtractorPhase3:
    """Phase 3 元数据提取器"""
    
    def __init__(self, qc_dir: str, para_file: str):
        self.qc_dir = Path(qc_dir)
        self.para_file = Path(para_file)
        self.metadata_list: List[DocumentMetadata] = []
        
    def compute_paper_hash(self, text: str) -> str:
        """计算文档内容哈希（稳定主键）"""
        # 使用前2000字符计算哈希
        sample = text[:2000].encode('utf-8')
        return hashlib.sha256(sample).hexdigest()[:16]
    
    def extract_year_from_path(self, md_path: Path) -> int:
        """从路径提取年份"""
        # 尝试从祖父目录名提取 (路径结构: .../2016/论文名/论文.md)
        if md_path.parent.parent:
            grandparent = md_path.parent.parent.name
            if grandparent.isdigit():
                year = int(grandparent)
                if 2000 <= year <= 2030:
                    return year
        
        # 尝试从父目录名提取
        parent = md_path.parent.name
        if parent.isdigit():
            year = int(parent)
            if 2000 <= year <= 2030:
                return year
        
        # 尝试从文件名提取
        stem = md_path.stem
        year_match = re.search(r'(20\d{2})', stem)
        if year_match:
            return int(year_match.group(1))
        
        return 0
    
    def extract_title(self, text: str) -> str:
        """提取论文标题"""
        lines = text.split('\n')
        
        # 查找第一个一级标题
        for line in lines[:50]:  # 只看前50行
            line = line.strip()
            
            # Markdown 一级标题
            if line.startswith('# '):
                title = line[2:].strip()
                if len(title) > 5 and len(title) < 100:
                    return title
            
            # 居中格式的标题（如硕士学位论文标题）
            if '硕士学位论文' in line or 'MBA论文' in line:
                # 下一行可能是标题
                continue
        
        # 从文件名推断
        return ""
    
    def extract_school(self, text: str) -> str:
        """提取学校"""
        patterns = [
            r'(\S+大学)',
            r'(\S+学院)',
            r'(\S+研究院)',
        ]
        
        # 看前100行
        for line in text.split('\n')[:100]:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    school = match.group(1)
                    # 过滤掉不可能是学校的
                    if len(school) > 4 and '专业' not in school and '班级' not in school:
                        return school
        
        return ""
    
    def extract_major(self, text: str) -> str:
        """提取专业"""
        # 查找专业/学科相关
        patterns = [
            r'专业\s*[:：]?\s*(\S+?)(?:\s|$)',
            r'学科\s*[:：]?\s*(\S+?)(?:\s|$)',
            r'(工商管理硕士|MBA|EMBA)',
            r'研究方向\s*[:：]?\s*(\S+?)(?:\s|$)',
        ]
        
        for line in text.split('\n')[:100]:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    return match.group(1).strip()
        
        return ""
    
    def extract_author(self, text: str) -> str:
        """提取作者"""
        patterns = [
            r'作者\s*[:：]?\s*(\S+?)(?:\s|$)',
            r'研究生\s*[:：]?\s*(\S+?)(?:\s|$)',
            r'姓名\s*[:：]?\s*(\S+?)(?:\s|$)',
        ]
        
        for line in text.split('\n')[:100]:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    author = match.group(1).strip()
                    if 2 <= len(author) <= 10:
                        return author
        
        return ""
    
    def count_chapters(self, text: str) -> int:
        """统计章节数"""
        chapter_patterns = [
            r'^第[一二三四五六七八九十\d]+章',
            r'^#{1,6}\s+',
        ]
        
        count = 0
        for line in text.split('\n'):
            for pattern in chapter_patterns:
                if re.match(pattern, line.strip()):
                    count += 1
                    break
        
        return count
    
    def extract_metadata(self, md_path: Path, paragraphs: List[Dict]) -> DocumentMetadata:
        """提取单个文档的元数据"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            text = ""
        
        doc_id = md_path.stem
        year = self.extract_year_from_path(md_path)
        title = self.extract_title(text)
        school = self.extract_school(text)
        major = self.extract_major(text)
        author = self.extract_author(text)
        
        # 计算哈希
        paper_hash = self.compute_paper_hash(text)
        source_doc_id = f"{year}_{doc_id}" if year else doc_id
        
        # 内容统计
        char_count = len(text)
        para_count = len(paragraphs)
        chapter_count = self.count_chapters(text)
        
        # 检查完整性
        missing_fields = []
        if not year or year == 0:
            missing_fields.append("year")
        if not title:
            missing_fields.append("title")
        if not school:
            missing_fields.append("school")
        if not major:
            missing_fields.append("major")
        if not author:
            missing_fields.append("author")
        
        has_complete_metadata = len(missing_fields) == 0
        
        relative_path = md_path.relative_to(self.qc_dir).as_posix()
        
        return DocumentMetadata(
            doc_id=doc_id,
            paper_hash=paper_hash,
            source_doc_id=source_doc_id,
            year=year,
            title=title,
            school=school,
            major=major,
            author=author,
            char_count=char_count,
            para_count=para_count,
            chapter_count=chapter_count,
            source_path=str(md_path),
            relative_path=relative_path,
            has_complete_metadata=has_complete_metadata,
            missing_fields=missing_fields
        )
    
    def run(self) -> List[DocumentMetadata]:
        """运行提取"""
        # 加载段落数据
        with open(self.para_file, 'r', encoding='utf-8') as f:
            all_paragraphs = json.load(f)
        
        # 按文档分组
        doc_paras = defaultdict(list)
        for para in all_paragraphs:
            doc_id = para['doc_id']
            doc_paras[doc_id].append(para)
        
        print(f"共 {len(doc_paras)} 个文档")
        
        # 获取所有 Markdown 文件
        md_files = list(self.qc_dir.rglob("*.md"))
        md_files = [f for f in md_files if "qc_" not in f.name and f.stem in doc_paras]
        
        print(f"找到 {len(md_files)} 个 Markdown 文件")
        
        # 提取元数据
        for md_path in md_files:
            doc_id = md_path.stem
            paragraphs = doc_paras.get(doc_id, [])
            
            metadata = self.extract_metadata(md_path, paragraphs)
            self.metadata_list.append(metadata)
        
        return self.metadata_list
    
    def save(self, output_dir: str):
        """保存元数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存完整元数据
        metadata_file = output_path / "documents_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_documents": len(self.metadata_list),
                "metadata": [asdict(m) for m in self.metadata_list]
            }, f, ensure_ascii=False, indent=2)
        
        print(f"元数据保存至: {metadata_file}")
        
        # 保存主键映射表（用于困难样本集）
        key_mapping = []
        for m in self.metadata_list:
            key_mapping.append({
                "paper_hash": m.paper_hash,
                "source_doc_id": m.source_doc_id,
                "doc_id": m.doc_id,
                "year": m.year,
                "has_complete_metadata": m.has_complete_metadata
            })
        
        key_file = output_path / "paper_key_mapping.json"
        with open(key_file, 'w', encoding='utf-8') as f:
            json.dump(key_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"主键映射保存至: {key_file}")
        
        # 保存统计
        complete_count = sum(1 for m in self.metadata_list if m.has_complete_metadata)
        year_stats = defaultdict(int)
        for m in self.metadata_list:
            year_stats[m.year] += 1
        
        stats_file = output_path / "metadata_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total": len(self.metadata_list),
                "complete_metadata": complete_count,
                "incomplete_metadata": len(self.metadata_list) - complete_count,
                "year_distribution": {str(k): v for k, v in sorted(year_stats.items())}
            }, f, ensure_ascii=False, indent=2)
        
        print(f"统计保存至: {stats_file}")
        print(f"\n元数据完整度: {complete_count}/{len(self.metadata_list)} ({complete_count/len(self.metadata_list)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="S3 Phase 3: 元数据提取")
    parser.add_argument("--qc_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/01_qc_output",
                        help="质控后 Markdown 目录")
    parser.add_argument("--para_file", type=str,
                        default="/home/aigc/aigc_checker/phase3/02_paragraphs/paragraphs_all.json",
                        help="段落文件")
    parser.add_argument("--output_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/03_metadata",
                        help="输出目录")
    
    args = parser.parse_args()
    
    extractor = MetadataExtractorPhase3(args.qc_dir, args.para_file)
    extractor.run()
    extractor.save(args.output_dir)


if __name__ == "__main__":
    main()
