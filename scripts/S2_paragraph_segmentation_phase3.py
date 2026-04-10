#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2 Phase 3: 段落切分（稳定类型版）

核心调整：
- 更严格地区分 heading / abstract / table_note / figure_note / formula_note / list_like / normal
- 明确增加 chapter_role 的稳定映射
- 对附录、公式说明、图表密集段做更保守处理
- 减少"模型学段落位置和结构模式"的机会

输入: phase3/01_qc_output/
输出: phase3/02_paragraphs/paragraphs_all.json
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class Paragraph:
    """段落数据结构（Phase 3 增强版）"""
    text: str
    doc_id: str
    year: int
    chapter: str
    para_index: int
    para_type: str       # heading/abstract/table_note/figure_note/formula_note/list_like/normal
    chapter_role: str    # intro/methodology/findings/discussion/conclusion/reference/appendix/other
    char_count: int
    word_count: int
    sentence_count: int
    has_citation: bool
    has_formula: bool
    has_table_ref: bool
    has_figure_ref: bool


class ParagraphSegmenterPhase3:
    """Phase 3 段落切分器（稳定类型版）"""
    
    def __init__(self):
        # 章节标题正则
        self.chapter_patterns = [
            (r'^#{1,6}\s+', 'heading'),
            (r'^第[一二三四五六七八九十\d]+章', 'heading'),
            (r'^第[一二三四五六七八九十\d]+节', 'heading'),
            (r'^第[一二三四五六七八九十\d]+篇', 'heading'),
            (r'^\d+\.\d+\s+[^\s]', 'heading'),
            (r'^\d+\.\d+\.\d+\s+[^\s]', 'heading'),
            (r'^\d+\.\d+\.\d+\.\d+\s+[^\s]', 'heading'),
        ]
        
        # 过滤模式（正文范围外）
        self.filter_patterns = [
            r'^\s*目录\s*$',
            r'^\s*图\s*\d+\s*[.。…]+\s*\d+$',
            r'^\s*表\s*\d+\s*[.。…]+\s*\d+$',
            r'^\s*作者简介',
            r'^\s*攻读学位期间',
        ]
        
        # chapter_role 关键词映射
        self.chapter_role_keywords = {
            "intro": ["绪论", "引言", "导论", "绪言", "研究背景", "研究意义", "问题提出"],
            "literature": ["文献综述", "理论基础", "相关研究", "国内外研究", "研究现状"],
            "methodology": ["研究方法", "研究设计", "数据", "样本", "问卷", "实证设计", "模型", "变量"],
            "findings": ["研究结果", "实证结果", "数据分析", "描述性统计", "回归结果", "假设检验"],
            "discussion": ["讨论", "结果讨论", "理论贡献", "实践启示", "管理启示"],
            "conclusion": ["结论", "总结", "研究结论", "主要发现"],
            "limitation": ["局限", "不足", "研究局限", "未来研究"],
            "reference": ["参考文献", "文献"],
            "appendix": ["附录", "附表", "附图", "调查问卷"],
            "acknowledgement": ["致谢", "感谢"],
            "abstract": ["摘要", "abstract"],
        }
    
    def is_chapter_header(self, text: str) -> Tuple[bool, str]:
        """判断是否为章节标题，返回(是否标题, 匹配模式)"""
        for pattern, ptype in self.chapter_patterns:
            if re.match(pattern, text.strip()):
                return True, ptype
        return False, ""
    
    def should_filter(self, text: str) -> bool:
        """判断是否应该过滤"""
        for pattern in self.filter_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        return False
    
    def detect_para_type(self, text: str, is_header: bool = False) -> str:
        """
        检测段落类型（Phase 3 增强版）
        """
        text_stripped = text.strip()
        
        # 标题类型
        if is_header:
            return "heading"
        
        # 摘要
        if re.match(r'^摘要\s*$', text_stripped, re.IGNORECASE):
            return "abstract"
        if re.match(r'^ABSTRACT\s*$', text_stripped, re.IGNORECASE):
            return "abstract"
        
        # 表格说明 - 更严格的模式
        if re.match(r'^表\s*[\d\-]+[\.、\s]', text_stripped):
            return "table_note"
        if re.match(r'^表\s*\d+\s+[^。]{1,30}$', text_stripped):
            return "table_note"
        
        # 图注 - 更严格的模式
        if re.match(r'^图\s*[\d\-]+[\.、\s]', text_stripped):
            return "figure_note"
        if re.match(r'^图\s*\d+\s+[^。]{1,30}$', text_stripped):
            return "figure_note"
        
        # 公式说明
        if re.match(r'^公式\s*[\d\(\[]', text_stripped):
            return "formula_note"
        if re.match(r'^式\s*[\(\[]\s*\d+', text_stripped):
            return "formula_note"
        if re.match(r'^\(\s*\d+\s*\)', text_stripped) and len(text_stripped) < 50:
            return "formula_note"
        
        # 列表型段落
        if re.match(r'^\d+、\s', text_stripped):
            return "list_like"
        if re.match(r'^\(\d+\)\s', text_stripped):
            return "list_like"
        if re.match(r'^\d+\.\s+\S', text_stripped):
            # 检查是否是章节编号（短）或列表项（长）
            if len(text_stripped) > 40:
                return "list_like"
            return "heading"
        if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]', text_stripped):
            return "list_like"
        if re.match(r'^[•·◆◇○●]', text_stripped):
            return "list_like"
        
        # 默认：常规正文
        return "normal"
    
    def detect_chapter_role(self, chapter_title: str, para_type: str) -> str:
        """
        检测章节角色（Phase 3 新增）
        """
        if para_type == "abstract":
            return "abstract"
        
        chapter_lower = chapter_title.lower()
        
        for role, keywords in self.chapter_role_keywords.items():
            for kw in keywords:
                if kw in chapter_lower:
                    return role
        
        return "other"
    
    def detect_features(self, text: str) -> Dict[str, bool]:
        """检测文本特征"""
        return {
            "has_citation": bool(re.search(r'\[\d+\]|\(\s*\d{4}\s*\)|et\s+al', text)),
            "has_formula": bool(re.search(r'[\$=∑∏∫√∂∆αβγδεθλμσπω]', text)) or '公式' in text,
            "has_table_ref": bool(re.search(r'表\s*\d+', text)),
            "has_figure_ref": bool(re.search(r'图\s*\d+', text)),
        }
    
    def count_sentences(self, text: str) -> int:
        """统计句子数"""
        # 简单的句子切分
        sentences = re.split(r'[。！？；\.!?]', text)
        return len([s for s in sentences if s.strip()])
    
    def segment_markdown(self, md_content: str, doc_id: str, year: int) -> List[Paragraph]:
        """
        切分 Markdown 为段落
        """
        paragraphs = []
        current_chapter = "未知章节"
        current_chapter_role = "other"
        para_index = 0
        
        lines = md_content.split('\n')
        current_para_lines = []
        
        def save_current_paragraph():
            """保存当前段落"""
            nonlocal para_index, current_chapter_role
            
            if not current_para_lines:
                return
            
            text = ' '.join(current_para_lines).strip()
            if not text or self.should_filter(text):
                current_para_lines.clear()
                return
            
            # 检测段落类型
            para_type = self.detect_para_type(text)
            
            # 极短噪声段合并
            if len(text) < 15 and paragraphs:
                paragraphs[-1].text += " " + text
                paragraphs[-1].char_count = len(paragraphs[-1].text)
                current_para_lines.clear()
                return
            
            # 检测特征
            features = self.detect_features(text)
            
            para_index += 1
            para = Paragraph(
                text=text,
                doc_id=doc_id,
                year=year,
                chapter=current_chapter,
                para_index=para_index,
                para_type=para_type,
                chapter_role=current_chapter_role,
                char_count=len(text),
                word_count=len([w for w in text if '\u4e00' <= w <= '\u9fff']),
                sentence_count=self.count_sentences(text),
                **features
            )
            paragraphs.append(para)
            current_para_lines.clear()
        
        for line in lines:
            line = line.strip()
            
            # 检测章节标题
            is_header, _ = self.is_chapter_header(line)
            if is_header:
                save_current_paragraph()
                current_chapter = line.lstrip('#').strip()
                current_chapter_role = self.detect_chapter_role(current_chapter, "heading")
                
                para_index += 1
                para = Paragraph(
                    text=line,
                    doc_id=doc_id,
                    year=year,
                    chapter=current_chapter,
                    para_index=para_index,
                    para_type="heading",
                    chapter_role=current_chapter_role,
                    char_count=len(line),
                    word_count=len(line),
                    sentence_count=0,
                    has_citation=False,
                    has_formula=False,
                    has_table_ref=False,
                    has_figure_ref=False
                )
                paragraphs.append(para)
                continue
            
            # 空行表示段落结束
            if not line:
                save_current_paragraph()
                continue
            
            current_para_lines.append(line)
        
        # 保存最后一个段落
        save_current_paragraph()
        
        return paragraphs


def process_markdown_file(md_path: Path, segmenter: ParagraphSegmenterPhase3) -> List[Paragraph]:
    """处理单个 Markdown 文件"""
    # 提取年份和文档ID
    year_str = md_path.parent.name
    year = int(year_str) if year_str.isdigit() else 0
    doc_id = md_path.stem
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    paragraphs = segmenter.segment_markdown(content, doc_id, year)
    return paragraphs


def main():
    parser = argparse.ArgumentParser(description="S2 Phase 3: 段落切分（稳定类型版）")
    parser.add_argument("--input_dir", type=str, 
                        default="/home/aigc/aigc_checker/phase3/01_qc_output",
                        help="输入目录")
    parser.add_argument("--output_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/02_paragraphs",
                        help="输出目录")
    parser.add_argument("--min_length", type=int, default=20,
                        help="最小段落长度")
    parser.add_argument("--max_length", type=int, default=2000,
                        help="最大段落长度")
    
    args = parser.parse_args()
    
    # 获取所有 Markdown 文件
    md_files = list(Path(args.input_dir).rglob("*.md"))
    print(f"找到 {len(md_files)} 个 Markdown 文件")
    
    # 初始化切分器
    segmenter = ParagraphSegmenterPhase3()
    
    # 处理每个文件
    all_paragraphs = []
    for i, md_file in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] 处理: {md_file}")
        try:
            paras = process_markdown_file(md_file, segmenter)
            # 过滤长度
            paras = [p for p in paras 
                     if args.min_length <= p.char_count <= args.max_length]
            all_paragraphs.extend(paras)
            print(f"  提取 {len(paras)} 个段落")
        except Exception as e:
            print(f"  错误: {e}")
    
    print(f"\n总计提取 {len(all_paragraphs)} 个段落")
    
    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "paragraphs_all.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(p) for p in all_paragraphs], f,
                  ensure_ascii=False, indent=2)
    
    print(f"结果保存至: {output_file}")
    
    # 统计
    type_stats = defaultdict(int)
    role_stats = defaultdict(int)
    year_stats = defaultdict(int)
    
    for p in all_paragraphs:
        type_stats[p.para_type] += 1
        role_stats[p.chapter_role] += 1
        year_stats[p.year] += 1
    
    print("\n段落类型分布:")
    for ptype, count in sorted(type_stats.items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count}")
    
    print("\n章节角色分布:")
    for role, count in sorted(role_stats.items(), key=lambda x: -x[1]):
        print(f"  {role}: {count}")
    
    print("\n年份分布:")
    for year in sorted(year_stats.keys()):
        print(f"  {year}: {year_stats[year]}")
    
    # 保存统计
    stats_file = output_dir / "segmentation_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_paragraphs": len(all_paragraphs),
            "type_distribution": dict(type_stats),
            "role_distribution": dict(role_stats),
            "year_distribution": {str(k): v for k, v in year_stats.items()}
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
