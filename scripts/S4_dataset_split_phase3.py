#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4 Phase 3: 数据集划分（双轨版）

核心调整：
- 保留时间切分思想
- 重新做 3 套集合：标准 train/dev/test + 困难样本 dev_hard/test_hard
- 标准集和困难集彻底隔离

时间切分：
- Train: 2015-2020
- Dev: 2021-2022
- Test: 2023-2024

困难样本构造：
- dev_hard: 从 dev 中选取难区分的样本
- test_hard: 从 test 中选取难区分的样本

输入: phase3/02_paragraphs/paragraphs_all.json, phase3/03_metadata/
输出: phase3/04_dataset_split/
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from datetime import datetime


class DatasetSplitterPhase3:
    """Phase 3 数据集划分器（双轨版）"""
    
    def __init__(self, para_file: str, metadata_file: str, output_dir: str, seed: int = 42):
        self.para_file = Path(para_file)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        random.seed(seed)
        
        # 时间切分规则
        self.year_split = {
            "train": (2015, 2020),
            "dev": (2021, 2022),
            "test": (2023, 2024),
        }
        
        # 困难样本比例
        self.hard_ratio = 0.3  # 30%作为困难样本
        
        # 数据容器
        self.all_paragraphs: List[Dict] = []
        self.metadata: List[Dict] = []
        self.doc_by_year: Dict[int, List[str]] = defaultdict(list)
        
    def load_data(self):
        """加载数据"""
        # 加载段落
        with open(self.para_file, 'r', encoding='utf-8') as f:
            self.all_paragraphs = json.load(f)
        
        # 加载元数据
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.metadata = data.get("metadata", [])
        
        # 按年份组织文档
        for m in self.metadata:
            year = m.get("year", 0)
            doc_id = m.get("doc_id", "")
            if year and doc_id:
                self.doc_by_year[year].append(doc_id)
        
        print(f"加载 {len(self.all_paragraphs)} 个段落")
        print(f"加载 {len(self.metadata)} 个文档元数据")
        print(f"年份分布: {dict(sorted(self.doc_by_year.items()))}")
    
    def get_docs_for_split(self, split_name: str) -> List[str]:
        """获取指定切分的文档列表"""
        year_range = self.year_split[split_name]
        docs = []
        for year in range(year_range[0], year_range[1] + 1):
            docs.extend(self.doc_by_year.get(year, []))
        return docs
    
    def filter_paragraphs_by_docs(self, paragraphs: List[Dict], docs: List[str]) -> List[Dict]:
        """按文档列表过滤段落"""
        doc_set = set(docs)
        return [p for p in paragraphs if p.get("doc_id", "") in doc_set]
    
    def select_hard_samples(self, paragraphs: List[Dict], ratio: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
        """
        选择困难样本
        
        困难样本标准：
        1. 长度适中的段落（50-300字符）- 太短太长的都容易区分
        2. 包含引用或专业术语的段落 - 更贴近真实学术写作
        3. 方法/讨论章节的段落 - 这些部分 AI 和人类写作更接近
        4. 列表或结构化段落 - AI 容易生成规整列表
        
        返回: (标准集段落, 困难集段落)
        """
        if not paragraphs:
            return [], []
        
        # 计算困难分数
        scored_paras = []
        for p in paragraphs:
            score = 0.0
            char_count = p.get("char_count", 0)
            
            # 长度适中加分（50-300字符）
            if 50 <= char_count <= 300:
                score += 2.0
            
            # 包含引用加分
            if p.get("has_citation", False):
                score += 1.5
            
            # 包含公式加分
            if p.get("has_formula", False):
                score += 1.0
            
            # 方法/讨论章节加分
            if p.get("chapter_role", "") in ["methodology", "discussion", "findings"]:
                score += 2.0
            
            # 常规正文加分（heading/list_like等容易区分）
            if p.get("para_type", "") == "normal":
                score += 1.0
            
            scored_paras.append((p, score))
        
        # 按分数排序
        scored_paras.sort(key=lambda x: -x[1])
        
        # 取前 ratio 作为困难样本候选
        hard_count = int(len(scored_paras) * ratio)
        hard_candidates = scored_paras[:hard_count]
        standard_candidates = scored_paras[hard_count:]
        
        # 从困难候选中按文档均匀采样，确保不把所有段落都选走
        doc_para_count = defaultdict(int)
        for p, _ in hard_candidates:
            doc_para_count[p["doc_id"]] += 1
        
        # 每个文档最多保留50%作为困难样本
        hard_paras = []
        standard_from_hard = []
        
        doc_hard_count = defaultdict(int)
        for p, score in hard_candidates:
            doc_id = p["doc_id"]
            total_doc_paras = doc_para_count[doc_id]
            
            # 如果已经选了超过50%，放入标准集
            if doc_hard_count[doc_id] >= total_doc_paras * 0.5:
                standard_from_hard.append(p)
            else:
                hard_paras.append(p)
                doc_hard_count[doc_id] += 1
        
        # 标准集 = 原来的标准候选 + 从困难候选退回的
        standard_paras = [p for p, _ in standard_candidates] + standard_from_hard
        
        return standard_paras, hard_paras
    
    def split_dataset(self):
        """执行数据集划分"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "splits": {}
        }
        
        # 标准集划分
        for split_name in ["train", "dev", "test"]:
            docs = self.get_docs_for_split(split_name)
            paragraphs = self.filter_paragraphs_by_docs(self.all_paragraphs, docs)
            
            # 保存标准集
            output_file = self.output_dir / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(paragraphs, f, ensure_ascii=False, indent=2)
            
            results["splits"][split_name] = {
                "doc_count": len(docs),
                "para_count": len(paragraphs),
                "year_range": self.year_split[split_name],
                "file": str(output_file)
            }
            
            print(f"{split_name}: {len(docs)} 文档, {len(paragraphs)} 段落")
        
        # 困难样本集
        # dev_hard
        dev_docs = self.get_docs_for_split("dev")
        dev_paras = self.filter_paragraphs_by_docs(self.all_paragraphs, dev_docs)
        dev_standard, dev_hard = self.select_hard_samples(dev_paras, self.hard_ratio)
        
        with open(self.output_dir / "dev_hard.json", 'w', encoding='utf-8') as f:
            json.dump(dev_hard, f, ensure_ascii=False, indent=2)
        
        # 更新 dev 标准集（移除困难样本）
        with open(self.output_dir / "dev.json", 'w', encoding='utf-8') as f:
            json.dump(dev_standard, f, ensure_ascii=False, indent=2)
        
        results["splits"]["dev_hard"] = {
            "doc_count": len(set(p["doc_id"] for p in dev_hard)),
            "para_count": len(dev_hard),
            "source": "dev",
            "file": str(self.output_dir / "dev_hard.json")
        }
        
        print(f"dev_hard: {results['splits']['dev_hard']['doc_count']} 文档, {len(dev_hard)} 段落")
        
        # test_hard
        test_docs = self.get_docs_for_split("test")
        test_paras = self.filter_paragraphs_by_docs(self.all_paragraphs, test_docs)
        test_standard, test_hard = self.select_hard_samples(test_paras, self.hard_ratio)
        
        with open(self.output_dir / "test_hard.json", 'w', encoding='utf-8') as f:
            json.dump(test_hard, f, ensure_ascii=False, indent=2)
        
        # 更新 test 标准集（移除困难样本）
        with open(self.output_dir / "test.json", 'w', encoding='utf-8') as f:
            json.dump(test_standard, f, ensure_ascii=False, indent=2)
        
        results["splits"]["test_hard"] = {
            "doc_count": len(set(p["doc_id"] for p in test_hard)),
            "para_count": len(test_hard),
            "source": "test",
            "file": str(self.output_dir / "test_hard.json")
        }
        
        print(f"test_hard: {results['splits']['test_hard']['doc_count']} 文档, {len(test_hard)} 段落")
        
        # 保存索引文件（用于快速查找）
        self._save_indices(results)
        
        # 保存统计
        with open(self.output_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n统计保存至: {self.output_dir / 'statistics.json'}")
    
    def _save_indices(self, results: Dict):
        """保存索引文件"""
        # 文档到切分的映射
        doc_to_split = {}
        for split_name in ["train", "dev", "test"]:
            docs = self.get_docs_for_split(split_name)
            for doc in docs:
                doc_to_split[doc] = split_name
        
        # 困难样本文档标记
        hard_docs = set()
        with open(self.output_dir / "dev_hard.json", 'r', encoding='utf-8') as f:
            dev_hard = json.load(f)
            for p in dev_hard:
                hard_docs.add((p["doc_id"], "dev_hard"))
        
        with open(self.output_dir / "test_hard.json", 'r', encoding='utf-8') as f:
            test_hard = json.load(f)
            for p in test_hard:
                hard_docs.add((p["doc_id"], "test_hard"))
        
        # 保存索引
        index = {
            "doc_to_split": doc_to_split,
            "hard_docs": list(hard_docs)
        }
        
        with open(self.output_dir / "split_index.json", 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        print(f"索引保存至: {self.output_dir / 'split_index.json'}")


def main():
    parser = argparse.ArgumentParser(description="S4 Phase 3: 数据集划分（双轨版）")
    parser.add_argument("--para_file", type=str,
                        default="/home/aigc/aigc_checker/phase3/02_paragraphs/paragraphs_all.json",
                        help="段落文件")
    parser.add_argument("--metadata_file", type=str,
                        default="/home/aigc/aigc_checker/phase3/03_metadata/documents_metadata.json",
                        help="元数据文件")
    parser.add_argument("--output_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/04_dataset_split",
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--hard_ratio", type=float, default=0.3,
                        help="困难样本比例")
    
    args = parser.parse_args()
    
    splitter = DatasetSplitterPhase3(
        para_file=args.para_file,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        seed=args.seed
    )
    splitter.hard_ratio = args.hard_ratio
    
    splitter.load_data()
    splitter.split_dataset()


if __name__ == "__main__":
    main()
