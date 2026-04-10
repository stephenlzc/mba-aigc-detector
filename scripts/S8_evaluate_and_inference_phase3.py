#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S8 Phase 3: 评估与推理（双轨评估版）

核心调整：
- 同时输出 standard_eval 和 hard_eval
- 单独输出困难样本集指标
- 增加过拟合风险提示

输入: phase3/06_features/, phase3/07_models/
输出: phase3/08_evaluation/
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from sklearn.metrics import (classification_report, roc_auc_score, 
                             precision_recall_curve, confusion_matrix)


class EvaluatorPhase3:
    """Phase 3 评估器"""
    
    def __init__(self, features_dir: str, models_dir: str, output_dir: str):
        self.features_dir = Path(features_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def load_features(self, split_name: str) -> tuple:
        """加载特征"""
        feature_file = self.features_dir / f"{split_name}_features.json"
        
        print(f"  加载: {feature_file.name}")
        
        with open(feature_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data.get("features", [])
        
        if not features:
            print(f"    ⚠️ 警告: 无特征数据")
            return np.array([]), np.array([]), []
        
        X = []
        y = []
        metadata = []
        
        # 检查第一个样本的字段，确定特征格式
        first_feat = features[0]
        
        if "combined_features" in first_feat:
            # 使用S6预计算的合并特征
            print(f"    使用 combined_features (维度: {len(first_feat['combined_features'])})")
            for feat in features:
                X.append(feat.get("combined_features", []))
                y.append(feat.get("label", 0))
                metadata.append({
                    "difficulty_level": feat.get("difficulty_level", "standard"),
                    "generation_path": feat.get("generation_path", "human"),
                    "para_type": feat.get("para_type", "normal"),
                })
        elif "bert_features" in first_feat and "stat_features" in first_feat:
            # 分别读取BERT特征和统计特征
            print(f"    使用 bert_features + stat_features")
            for feat in features:
                bert_feat = feat.get("bert_features", [])
                stat_feat = feat.get("stat_features", [])
                combined = bert_feat + stat_feat
                X.append(combined)
                y.append(feat.get("label", 0))
                metadata.append({
                    "difficulty_level": feat.get("difficulty_level", "standard"),
                    "generation_path": feat.get("generation_path", "human"),
                    "para_type": feat.get("para_type", "normal"),
                })
        else:
            # 回退到旧格式
            print(f"    使用旧格式特征")
            for feat in features:
                bert_feat = feat.get("bert_features", [])
                stat_feat = [
                    feat.get("char_count", 0),
                    feat.get("sentence_count", 0),
                    feat.get("chinese_ratio", 0),
                    feat.get("comma_count", 0),
                    feat.get("period_count", 0),
                    feat.get("vocab_diversity", 0),
                ]
                combined = bert_feat + stat_feat
                X.append(combined)
                y.append(feat.get("label", 0))
                metadata.append({
                    "difficulty_level": feat.get("difficulty_level", "standard"),
                    "generation_path": feat.get("generation_path", "human"),
                    "para_type": feat.get("para_type", "normal"),
                })
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"    样本数: {len(X_array)}, 特征维度: {X_array.shape[1] if len(X_array) > 0 else 0}")
        
        return X_array, y_array, metadata
    
    def load_model(self, model_name: str):
        """加载模型"""
        model_file = self.models_dir / f"{model_name}_model.pkl"
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    
    def evaluate_split(self, model, X: np.ndarray, y: np.ndarray, 
                       metadata: List[Dict], split_name: str) -> Dict:
        """评估单个切分"""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # 基础指标
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        results = {
            "split": split_name,
            "total_samples": len(y),
            "positive_samples": int(sum(y)),
            "negative_samples": int(len(y) - sum(y)),
            "precision": report["1"]["precision"] if "1" in report else 0,
            "recall": report["1"]["recall"] if "1" in report else 0,
            "f1": report["1"]["f1-score"] if "1" in report else 0,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
        }
        
        # AUC
        if len(set(y)) > 1:
            results["auc"] = roc_auc_score(y, y_prob)
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        results["confusion_matrix"] = cm.tolist()
        
        # 按难度分层
        standard_mask = [m.get("difficulty_level") == "standard" for m in metadata]
        hard_mask = [m.get("difficulty_level") == "hard" for m in metadata]
        
        if any(standard_mask):
            y_std = y[standard_mask]
            y_prob_std = y_prob[standard_mask]
            if len(set(y_std)) > 1:
                results["standard_auc"] = roc_auc_score(y_std, y_prob_std)
        
        if any(hard_mask):
            y_hard = y[hard_mask]
            y_prob_hard = y_prob[hard_mask]
            y_pred_hard = y_pred[hard_mask]
            
            if len(set(y_hard)) > 1:
                results["hard_auc"] = roc_auc_score(y_hard, y_prob_hard)
                results["hard_precision"] = np.mean([
                    1 if (y_hard[i] == 1 and y_pred_hard[i] == 1) else 0
                    for i in range(len(y_hard)) if y_hard[i] == 1
                ]) if sum(y_hard) > 0 else 0
                results["hard_recall"] = np.mean([
                    1 if y_pred_hard[i] == 1 else 0
                    for i in range(len(y_hard)) if y_hard[i] == 1
                ]) if sum(y_hard) > 0 else 0
        
        # 计算 uncertain 比例（概率在 0.4-0.6 之间）
        uncertain_mask = (y_prob >= 0.4) & (y_prob <= 0.6)
        results["uncertain_ratio"] = np.mean(uncertain_mask)
        
        return results
    
    def detect_overfitting(self, standard_results: Dict, hard_results: Dict) -> str:
        """检测过拟合风险"""
        std_f1 = standard_results.get("f1", 0)
        hard_f1 = hard_results.get("f1", 0)
        
        gap = std_f1 - hard_f1
        
        if gap > 0.2:
            return f"高风险：标准集F1({std_f1:.3f})远高于困难集F1({hard_f1:.3f})，差距{gap:.3f}"
        elif gap > 0.1:
            return f"中风险：标准集F1({std_f1:.3f})高于困难集F1({hard_f1:.3f})，差距{gap:.3f}"
        else:
            return "低风险：标准集与困难集性能接近"
    
    def run(self):
        """运行评估"""
        print("Phase 3 双轨评估")
        print("=" * 60)
        
        # 加载模型
        models = {
            "gbdt": self.load_model("gbdt"),
            "rf": self.load_model("rf"),
            "lr": self.load_model("lr"),
        }
        
        # 评估各个切分
        splits = ["dev", "test", "dev_hard", "test_hard"]
        
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\n评估模型: {model_name}")
            print("-" * 60)
            
            model_results = {}
            
            for split_name in splits:
                try:
                    X, y, metadata = self.load_features(split_name)
                    
                    if len(X) == 0:
                        print(f"  {split_name}: 无数据")
                        continue
                    
                    results = self.evaluate_split(model, X, y, metadata, split_name)
                    model_results[split_name] = results
                    
                    print(f"  {split_name}:")
                    print(f"    F1: {results['f1']:.3f}, AUC: {results.get('auc', 0):.3f}")
                    if "hard_auc" in results:
                        print(f"    Hard AUC: {results['hard_auc']:.3f}")
                    print(f"    Uncertain: {results['uncertain_ratio']:.1%}")
                    
                except Exception as e:
                    print(f"  {split_name}: 错误 - {e}")
            
            all_results[model_name] = model_results
            
            # 过拟合检测
            if "dev" in model_results and "dev_hard" in model_results:
                risk = self.detect_overfitting(
                    model_results["dev"], 
                    model_results["dev_hard"]
                )
                print(f"  过拟合风险: {risk}")
        
        # 保存结果
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": all_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估结果保存至: {results_file}")
        
        # 生成摘要报告
        self.generate_summary(all_results)
    
    def generate_summary(self, all_results: Dict):
        """生成摘要报告"""
        summary_file = self.output_dir / "evaluation_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Phase 3 评估摘要\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, model_results in all_results.items():
                f.write(f"模型: {model_name}\n")
                f.write("-" * 40 + "\n")
                
                for split_name, results in model_results.items():
                    f.write(f"  {split_name}:\n")
                    f.write(f"    F1: {results['f1']:.3f}\n")
                    f.write(f"    Precision: {results['precision']:.3f}\n")
                    f.write(f"    Recall: {results['recall']:.3f}\n")
                    f.write(f"    AUC: {results.get('auc', 0):.3f}\n")
                    if "hard_auc" in results:
                        f.write(f"    Hard AUC: {results['hard_auc']:.3f}\n")
                    f.write(f"    Uncertain Ratio: {results['uncertain_ratio']:.1%}\n")
                    f.write("\n")
                
                f.write("\n")
        
        print(f"摘要报告保存至: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="S8 Phase 3: 评估与推理")
    parser.add_argument("--features_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/06_features",
                        help="特征目录")
    parser.add_argument("--models_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/07_models",
                        help="模型目录")
    parser.add_argument("--output_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/08_evaluation",
                        help="输出目录")
    
    args = parser.parse_args()
    
    evaluator = EvaluatorPhase3(args.features_dir, args.models_dir, args.output_dir)
    evaluator.run()


if __name__ == "__main__":
    main()
