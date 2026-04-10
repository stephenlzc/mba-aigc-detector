#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S6 Phase 3: 特征工程（完整版 - BERT + 统计特征）

核心特性：
- 使用 roBERT_model 提取 BERT 特征（768维 CLS token）
- 结合统计特征
- 每50分钟 Server酱汇报进度
- 支持 GPU 加速
"""

import os
import re
import json
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import requests
from multiprocessing import Value as MPValue

# 加载 .env
load_dotenv("/home/aigc/aigc_checker/.env")


# ============ 配置 ============

REPORT_INTERVAL = 3000  # 50分钟汇报一次
BERT_BATCH_SIZE = 32    # GPU batch size
CPU_BATCH_SIZE = 8      # CPU batch size


# ============ Progress Reporter ============

class ProgressReporter(threading.Thread):
    """后台进度汇报线程，每50分钟汇报一次"""
    
    def __init__(self, processed_count: MPValue, total_samples: int, task_name: str):
        super().__init__(daemon=True)
        self.processed_count = processed_count
        self.total_samples = total_samples
        self.task_name = task_name
        self.start_time = time.time()
        self.last_report_time = 0
        self.report_interval = REPORT_INTERVAL
        self.serverchan_key = os.getenv("SERVERCHAN_SENDKEY", "")
        self.running = True
        
    def run(self):
        """后台线程运行"""
        while self.running:
            time.sleep(60)  # 每分钟检查一次
            elapsed = time.time() - self.start_time
            
            if elapsed - self.last_report_time >= self.report_interval:
                self.report()
                self.last_report_time = elapsed
    
    def stop(self):
        """停止汇报线程"""
        self.running = False
    
    def report(self):
        """发送进度汇报到 Server酱"""
        completed = self.processed_count.value
        remaining = max(0, self.total_samples - completed)
        elapsed = time.time() - self.start_time
        avg_time = elapsed / max(completed, 1)
        eta_seconds = avg_time * remaining
        
        # 构建汇报内容
        title = f"Phase3 {self.task_name} 进度汇报"
        content = f"""## {self.task_name} 进度

- **当前时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **已用时间**: {int(elapsed // 3600)}小时{int((elapsed % 3600) // 60)}分钟
- **已完成**: {completed} / {self.total_samples} ({completed/max(self.total_samples,1)*100:.1f}%)
- **剩余**: {remaining} 个
- **平均速度**: {avg_time:.2f}秒/样本
- **预计剩余时间**: {int(eta_seconds // 3600)}小时{int((eta_seconds % 3600) // 60)}分钟
- **使用设备**: {"GPU" if torch.cuda.is_available() else "CPU"}
"""
        
        # 发送 Server酱
        if self.serverchan_key:
            try:
                url = f"https://sctapi.ftqq.com/{self.serverchan_key}.send"
                resp = requests.post(url, data={
                    "title": title,
                    "desp": content
                }, timeout=30)
                print(f"[ProgressReporter] Server酱发送状态: {resp.status_code}")
            except Exception as e:
                print(f"[ProgressReporter] Server酱发送失败: {e}")
        
        # 同时写入日志
        log_file = Path("/home/aigc/aigc_checker/phase3/08_logs/progress_reports.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {title}\n{content}\n\n")
        
        print(f"[ProgressReporter] {title} - 已完成 {completed}/{self.total_samples}")


# ============ BERT Feature Extractor ============

class BERTFeatureExtractor:
    """BERT特征提取器"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BERT] 加载模型: {model_path}")
        print(f"[BERT] 使用设备: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        self.batch_size = BERT_BATCH_SIZE if torch.cuda.is_available() else CPU_BATCH_SIZE
        print(f"[BERT] Batch size: {self.batch_size}")
    
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """批量提取 BERT 特征（CLS token）"""
        all_features = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 提取 CLS token (768维)
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_features.append(features)
        
        return np.vstack(all_features)


# ============ Field Validator ============

class FieldValidator:
    """字段验证器"""
    
    REQUIRED_VARIANT_FIELDS = [
        'variant_text', 'doc_id', 'year', 'para_type', 
        'chapter_role', 'variant_type'
    ]
    
    @classmethod
    def validate_variant(cls, var: Dict, file_name: str, index: int) -> Tuple[bool, str]:
        """验证变体记录，返回(是否有效, 错误信息)"""
        missing_fields = []
        for field in cls.REQUIRED_VARIANT_FIELDS:
            if field not in var or var[field] is None or var[field] == "":
                missing_fields.append(field)
        
        if missing_fields:
            return False, f"{file_name}[{index}]: 缺少字段 {missing_fields}"
        
        # 检查year是否为0
        if var.get('year', 0) == 0:
            return False, f"{file_name}[{index}]: year=0"
        
        # 检查variant_text长度
        text = var.get('variant_text', '')
        if len(text) < 50:
            return False, f"{file_name}[{index}]: 文本太短({len(text)}字符)"
        
        return True, ""


# ============ Statistical Feature Extractor ============

class StatisticalFeatureExtractor:
    """统计特征提取器"""
    
    def extract(self, text: str) -> Dict:
        """提取统计特征"""
        features = {}
        
        # 基本统计
        features["char_count"] = len(text)
        features["sentence_count"] = len(re.split(r'[。！？；\.!?]', text))
        
        # 中文比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        features["chinese_ratio"] = chinese_chars / len(text) if text else 0
        
        # 英文比例
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        features["english_ratio"] = english_chars / len(text) if text else 0
        
        # 数字比例
        digits = len(re.findall(r'\d', text))
        features["digit_ratio"] = digits / len(text) if text else 0
        
        # 标点统计
        features["comma_count"] = text.count('，') + text.count(',')
        features["period_count"] = text.count('。') + text.count('.')
        features["semicolon_count"] = text.count('；') + text.count(';')
        features["colon_count"] = text.count('：') + text.count(':')
        
        # 词汇多样性
        words = re.findall(r'[\u4e00-\u9fff]', text)
        unique_words = len(set(words))
        features["vocab_diversity"] = unique_words / len(words) if words else 0
        
        # 平均句子长度
        sentences = [s for s in re.split(r'[。！？；\.!?]', text) if s.strip()]
        features["avg_sentence_length"] = np.mean([len(s) for s in sentences]) if sentences else 0
        
        # 特殊模式
        features["has_citation"] = 1 if re.search(r'\[\d+\]', text) else 0
        features["has_formula"] = 1 if re.search(r'[=∑∏∫√∂]', text) else 0
        features["has_table_ref"] = 1 if re.search(r'表\s*\d+', text) else 0
        features["has_figure_ref"] = 1 if re.search(r'图\s*\d+', text) else 0
        
        return features


# ============ Feature Engineering ============

class FeatureEngineer:
    """特征工程主类"""
    
    def __init__(self, model_path: str, output_dir: str):
        self.bert_extractor = BERTFeatureExtractor(model_path)
        self.stat_extractor = StatisticalFeatureExtractor()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_features(self, features_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List]:
        """处理特征列表，返回 (X, y, metadata)"""
        X_bert = []
        X_stat = []
        y = []
        metadata = []
        
        texts = [f.get("text", "") for f in features_list]
        
        print(f"[S6] 提取 BERT 特征（{len(texts)} 样本）...")
        bert_features = self.bert_extractor.extract_batch(texts)
        
        for i, feat in enumerate(features_list):
            # BERT 特征 (768维)
            bert_feat = bert_features[i].tolist()
            
            # 统计特征
            text = feat.get("text", "")
            stat_feat_dict = self.stat_extractor.extract(text)
            stat_feat = [
                stat_feat_dict["char_count"],
                stat_feat_dict["sentence_count"],
                stat_feat_dict["chinese_ratio"],
                stat_feat_dict["english_ratio"],
                stat_feat_dict["digit_ratio"],
                stat_feat_dict["comma_count"],
                stat_feat_dict["period_count"],
                stat_feat_dict["vocab_diversity"],
                stat_feat_dict["avg_sentence_length"],
                stat_feat_dict["has_citation"],
                stat_feat_dict["has_formula"],
                stat_feat_dict["has_table_ref"],
                stat_feat_dict["has_figure_ref"],
            ]
            
            # 合并特征
            combined = bert_feat + stat_feat
            X_bert.append(combined)
            y.append(feat.get("label", 0))
            metadata.append({
                "difficulty_level": feat.get("difficulty_level", "standard"),
                "generation_path": feat.get("generation_path", "human"),
                "para_type": feat.get("para_type", "normal"),
            })
        
        return np.array(X_bert), np.array(y), metadata
    
    def get_variant_difficulty(self, variant_type: str) -> str:
        """根据变体类型判断难度等级"""
        # hard变体类型
        hard_variants = ['hard_human']
        return "hard" if variant_type in hard_variants else "standard"
    
    def process_split(self, split_file: Path, variant_files: List[Path],
                      is_hard: bool = False, progress_counter: MPValue = None) -> Dict:
        """处理单个切分"""
        print(f"\n{'='*60}")
        print(f"[S6] 处理: {split_file.name}")
        print(f"{'='*60}")
        
        # 加载段落
        with open(split_file, 'r', encoding='utf-8') as f:
            paragraphs = json.load(f)
        
        # 过滤有效段落
        valid_paras = [p for p in paragraphs if 50 <= len(p.get("text", "")) <= 500]
        print(f"[S6] 人类段落: {len(valid_paras)} 条")
        
        features_list = []
        
        for para in valid_paras:
            text = para.get("text", "")
            
            feature_record = {
                "text": text,
                "doc_id": para.get("doc_id"),
                "para_index": para.get("para_index"),
                "year": para.get("year"),
                "para_type": para.get("para_type"),
                "chapter_role": para.get("chapter_role"),
                "label": 0,  # 人类写作
                "difficulty_level": "hard" if is_hard else "standard",
                "generation_path": "human",
            }
            
            features_list.append(feature_record)
            
            if progress_counter:
                progress_counter.value += 1
        
        # 处理变体（AI 生成）
        print(f"\n[S6] 处理变体文件 ({len(variant_files)} 个)...")
        variant_stats = {}
        
        for var_file in variant_files:
            if not var_file.exists():
                print(f"  ⚠️ 跳过: {var_file.name} (不存在)")
                continue
            
            # 解析变体类型
            # 文件名格式: {split}_{variant_type}_variants.json
            parts = var_file.stem.split('_')
            if len(parts) >= 3 and parts[-1] == 'variants':
                variant_type = '_'.join(parts[1:-1])  # 去掉split前缀和variants后缀
            else:
                variant_type = "unknown"
            
            print(f"\n  📄 {var_file.name} (类型: {variant_type})")
            
            try:
                with open(var_file, 'r', encoding='utf-8') as f:
                    variants = json.load(f)
            except json.JSONDecodeError as e:
                print(f"    ❌ JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"    ❌ 读取错误: {e}")
                continue
            
            valid_count = 0
            invalid_count = 0
            
            for idx, var in enumerate(variants):
                # 字段验证
                is_valid, error_msg = FieldValidator.validate_variant(var, var_file.name, idx)
                if not is_valid:
                    if invalid_count < 3:  # 只显示前3个错误
                        print(f"    ⚠️ {error_msg}")
                    invalid_count += 1
                    continue
                
                text = var.get("variant_text", "")
                
                feature_record = {
                    "text": text,
                    "doc_id": var.get("doc_id", "unknown"),
                    "para_index": -1,
                    "year": var.get("year", 0),
                    "para_type": var.get("para_type", "normal"),
                    "chapter_role": var.get("chapter_role", "other"),
                    "label": 1,  # AI 生成
                    "difficulty_level": self.get_variant_difficulty(variant_type),
                    "generation_path": variant_type,
                }
                
                features_list.append(feature_record)
                valid_count += 1
                
                if progress_counter:
                    progress_counter.value += 1
            
            variant_stats[variant_type] = {
                'total': len(variants),
                'valid': valid_count,
                'invalid': invalid_count
            }
            print(f"    ✅ 有效: {valid_count}/{len(variants)} 条")
        
        # 打印汇总
        print(f"\n{'-'*60}")
        print(f"[S6] 汇总:")
        print(f"  人类: {len(valid_paras)} 条")
        ai_total = sum(s['valid'] for s in variant_stats.values())
        print(f"  AI: {ai_total} 条")
        for vt, stats in sorted(variant_stats.items()):
            print(f"    - {vt}: {stats['valid']} 条")
        print(f"  总计: {len(features_list)} 条")
        print(f"{'-'*60}")
        
        # 提取特征
        print(f"\n[S6] 开始提取特征...")
        X, y, metadata = self.process_features(features_list)
        
        return {
            "split_name": split_file.stem,
            "total_samples": len(features_list),
            "human_samples": sum(1 for f in features_list if f["label"] == 0),
            "ai_samples": sum(1 for f in features_list if f["label"] == 1),
            "variant_breakdown": variant_stats,
            "features": [
                {
                    "doc_id": f.get("doc_id"),
                    "para_index": f.get("para_index"),
                    "year": f.get("year"),
                    "para_type": f.get("para_type"),
                    "chapter_role": f.get("chapter_role"),
                    "label": f.get("label"),
                    "difficulty_level": f.get("difficulty_level"),
                    "generation_path": f.get("generation_path"),
                    "bert_features": X[i][:768].tolist(),  # 前768维是BERT
                    "stat_features": X[i][768:].tolist(),  # 后面是统计特征
                    "combined_features": X[i].tolist(),     # 完整特征
                }
                for i, f in enumerate(features_list)
            ]
        }
    
    def run(self, split_dir: str, variants_dir: str):
        """运行特征工程"""
        split_dir = Path(split_dir)
        variants_dir = Path(variants_dir)
        
        # 计算总样本数
        total_samples = 0
        for split_name in ["train", "dev", "test", "dev_hard", "test_hard"]:
            split_file = split_dir / f"{split_name}.json"
            if split_file.exists():
                with open(split_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_samples += len([p for p in data if 50 <= len(p.get("text", "")) <= 500])
        
        # 初始化进度计数器和 reporter
        progress_counter = MPValue('i', 0)
        reporter = ProgressReporter(progress_counter, total_samples, "S6")
        reporter.start()
        
        try:
            # 处理各个切分
            splits = [
                ("train", False),
                ("dev", False),
                ("test", False),
                ("dev_hard", True),
                ("test_hard", True),
            ]
            
            for split_name, is_hard in splits:
                split_file = split_dir / f"{split_name}.json"
                
                if not split_file.exists():
                    print(f"[S6] 跳过: {split_name} (文件不存在)")
                    continue
                
                # 查找对应的变体文件
                variant_files = list(variants_dir.glob(f"{split_name}_*_variants.json"))
                
                result = self.process_split(split_file, variant_files, is_hard, progress_counter)
                
                # 保存
                output_file = self.output_dir / f"{split_name}_features.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"[S6] 保存: {output_file}")
                print(f"[S6] 总计: {result['total_samples']}, "
                      f"人类: {result['human_samples']}, "
                      f"AI: {result['ai_samples']}")
            
            # 发送最终汇报
            reporter.report()
            
        finally:
            # 停止 reporter
            reporter.stop()
            reporter.join(timeout=5)


def print_summary_report(results: List[Dict]):
    """打印最终汇总报告"""
    print("\n" + "=" * 70)
    print("📊 S6 特征工程 - 最终汇总报告")
    print("=" * 70)
    
    total_human = 0
    total_ai = 0
    total_samples = 0
    
    variant_totals = {}
    
    for result in results:
        split_name = result['split_name']
        human = result['human_samples']
        ai = result['ai_samples']
        total = result['total_samples']
        
        total_human += human
        total_ai += ai
        total_samples += total
        
        print(f"\n【{split_name}】")
        print(f"  总计: {total} 条")
        print(f"  人类: {human} 条")
        print(f"  AI: {ai} 条")
        
        # 变体细分
        if 'variant_breakdown' in result:
            print(f"  变体细分:")
            for vt, stats in sorted(result['variant_breakdown'].items()):
                print(f"    - {vt}: {stats['valid']} 条")
                variant_totals[vt] = variant_totals.get(vt, 0) + stats['valid']
    
    print("\n" + "-" * 70)
    print("📈 总体统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  人类样本: {total_human} ({total_human/total_samples*100:.1f}%)")
    print(f"  AI样本: {total_ai} ({total_ai/total_samples*100:.1f}%)")
    
    if variant_totals:
        print(f"\n  各变体类型统计:")
        for vt, count in sorted(variant_totals.items(), key=lambda x: -x[1]):
            print(f"    - {vt}: {count} 条")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="S6 Phase 3: 特征工程（完整版）")
    parser.add_argument("--model_path", type=str,
                        default="/home/aigc/aigc_checker/roBERT_model",
                        help="BERT 模型路径")
    parser.add_argument("--split_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/04_dataset_split",
                        help="数据集切分目录")
    parser.add_argument("--variants_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/05_ai_variants",
                        help="变体目录")
    parser.add_argument("--output_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/06_features",
                        help="输出目录")
    parser.add_argument("--skip_validation", action="store_true",
                        help="跳过字段验证（加快速度）")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("S6 Phase 3: 特征工程（完整版 - BERT + 统计特征）")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  数据切分: {args.split_dir}")
    print(f"  变体目录: {args.variants_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  字段验证: {'跳过' if args.skip_validation else '启用'}")
    print(f"  设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # 检查变体文件
    variants_dir = Path(args.variants_dir)
    variant_files = list(variants_dir.glob("*_variants.json"))
    print(f"\n发现 {len(variant_files)} 个变体文件:")
    for vf in sorted(variant_files):
        try:
            with open(vf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  - {vf.name}: {len(data)} 条")
        except Exception as e:
            print(f"  - {vf.name}: 读取错误 ({e})")
    
    print(f"\n{'='*70}\n")
    
    engineer = FeatureEngineer(args.model_path, args.output_dir)
    
    # 收集结果用于汇总报告
    results = []
    split_dir = Path(args.split_dir)
    variants_dir = Path(args.variants_dir)
    
    # 计算总样本数
    total_samples = 0
    for split_name in ["train", "dev", "test", "dev_hard", "test_hard"]:
        split_file = split_dir / f"{split_name}.json"
        if split_file.exists():
            with open(split_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_samples += len([p for p in data if 50 <= len(p.get("text", "")) <= 500])
    
    # 加上变体样本数
    for vf in variants_dir.glob("*_variants.json"):
        try:
            with open(vf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                total_samples += len(data)
        except:
            pass
    
    # 初始化进度计数器和 reporter
    progress_counter = MPValue('i', 0)
    reporter = ProgressReporter(progress_counter, total_samples, "S6")
    reporter.start()
    
    try:
        # 处理各个切分
        splits = [
            ("train", False),
            ("dev", False),
            ("test", False),
            ("dev_hard", True),
            ("test_hard", True),
        ]
        
        for split_name, is_hard in splits:
            split_file = split_dir / f"{split_name}.json"
            
            if not split_file.exists():
                print(f"\n[S6] 跳过: {split_name} (文件不存在)")
                continue
            
            # 查找对应的变体文件
            variant_files = list(variants_dir.glob(f"{split_name}_*_variants.json"))
            
            result = engineer.process_split(split_file, variant_files, is_hard, progress_counter)
            results.append(result)
            
            # 保存
            output_file = Path(args.output_dir) / f"{split_name}_features.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n[S6] 保存: {output_file}")
        
        # 发送最终汇报
        reporter.report()
        
        # 打印汇总报告
        print_summary_report(results)
        
    finally:
        # 停止 reporter
        reporter.stop()
        reporter.join(timeout=5)
    
    print("\n[S6] 完成！")


if __name__ == "__main__":
    main()
