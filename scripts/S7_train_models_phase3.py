#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7 Phase 3: 模型训练（双轨训练版）

核心调整：
- 标准正样本训练 + 困难正样本微调
- 双轨评估：standard_dev + hard_dev
- 模型选择看困难集指标

输入: phase3/06_features/
输出: phase3/07_models/
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV


class ModelTrainerPhase3:
    """Phase 3 模型训练器"""
    
    def __init__(self, features_dir: str, output_dir: str):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.results = {}
    
    def load_features(self, split_name: str) -> Tuple[np.ndarray, np.ndarray, List]:
        """加载特征"""
        feature_file = self.features_dir / f"{split_name}_features.json"
        
        print(f"  加载: {feature_file}")
        
        with open(feature_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        features = data.get("features", [])
        
        if not features:
            print(f"  ⚠️  警告: {feature_file} 中没有特征数据")
            return np.array([]), np.array([]), []
        
        X = []
        y = []
        metadata = []
        
        # 检查第一个样本的字段，确定特征格式
        first_feat = features[0]
        
        if "combined_features" in first_feat:
            # 使用S6预计算的合并特征（推荐）
            print(f"    使用 combined_features (维度: {len(first_feat['combined_features'])})")
            for feat in features:
                X.append(feat.get("combined_features", []))
                y.append(feat.get("label", 0))
                metadata.append({
                    "difficulty_level": feat.get("difficulty_level", "standard"),
                    "generation_path": feat.get("generation_path", "human"),
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
                })
        else:
            # 回退到旧格式（兼容旧数据）
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
                })
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"    样本数: {len(X_array)}, 特征维度: {X_array.shape[1] if len(X_array) > 0 else 0}")
        
        return X_array, y_array, metadata
    
    def train_baseline(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """训练基线模型"""
        print("训练基线模型...")
        
        models = {
            "gbdt": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            "rf": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            "lr": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
        }
        
        trained = {}
        for name, model in models.items():
            print(f"  训练 {name}...")
            model.fit(X_train, y_train)
            trained[name] = model
        
        return trained
    
    def evaluate(self, model, X: np.ndarray, y: np.ndarray, 
                 metadata: List[Dict], split_name: str) -> Dict:
        """评估模型"""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        # 基础指标
        report = classification_report(y, y_pred, output_dict=True)
        auc = roc_auc_score(y, y_prob)
        
        # 按难度分层评估
        standard_mask = [m.get("difficulty_level") == "standard" for m in metadata]
        hard_mask = [m.get("difficulty_level") == "hard" for m in metadata]
        
        results = {
            "split": split_name,
            "overall": {
                "precision": report["1"]["precision"] if "1" in report else 0,
                "recall": report["1"]["recall"] if "1" in report else 0,
                "f1": report["1"]["f1-score"] if "1" in report else 0,
                "auc": auc,
            },
        }
        
        # 困难样本指标
        if any(hard_mask):
            y_hard = y[hard_mask]
            y_prob_hard = y_prob[hard_mask]
            if len(set(y_hard)) > 1:
                results["hard"] = {
                    "auc": roc_auc_score(y_hard, y_prob_hard),
                }
        
        return results
    
    def run(self):
        """运行训练"""
        print("=" * 60)
        print("S7 Phase 3: 模型训练")
        print("=" * 60)
        
        # 加载数据
        print("\n【1/3】加载训练数据...")
        X_train, y_train, meta_train = self.load_features("train")
        if len(X_train) == 0:
            print("❌ 错误: 训练数据为空")
            return
        
        print("\n【2/3】加载验证数据...")
        X_dev, y_dev, meta_dev = self.load_features("dev")
        X_dev_hard, y_dev_hard, meta_dev_hard = self.load_features("dev_hard")
        
        if len(X_dev) == 0:
            print("⚠️ 警告: 标准验证集为空")
        if len(X_dev_hard) == 0:
            print("⚠️ 警告: 困难验证集为空")
        
        # 合并 dev 用于评估
        X_dev_all = np.vstack([X_dev, X_dev_hard])
        y_dev_all = np.concatenate([y_dev, y_dev_hard])
        meta_dev_all = meta_dev + meta_dev_hard
        
        # 训练基线模型
        print("\n【3/3】训练基线模型...")
        models = self.train_baseline(X_train, y_train)
        
        # 评估
        print("\n" + "=" * 60)
        print("模型评估结果")
        print("=" * 60)
        
        best_model = None
        best_f1 = 0
        
        for name, model in models.items():
            print(f"\n📊 {name.upper()}:")
            
            # 标准 dev
            dev_results = self.evaluate(model, X_dev, y_dev, meta_dev, "dev")
            print(f"  标准Dev  - P: {dev_results['overall']['precision']:.3f}, "
                  f"R: {dev_results['overall']['recall']:.3f}, "
                  f"F1: {dev_results['overall']['f1']:.3f}, "
                  f"AUC: {dev_results['overall']['auc']:.3f}")
            
            # 困难 dev
            if len(X_dev_hard) > 0:
                dev_hard_results = self.evaluate(model, X_dev_hard, y_dev_hard, 
                                                 meta_dev_hard, "dev_hard")
                print(f"  困难Dev  - P: {dev_hard_results['overall']['precision']:.3f}, "
                      f"R: {dev_hard_results['overall']['recall']:.3f}, "
                      f"F1: {dev_hard_results['overall']['f1']:.3f}, "
                      f"AUC: {dev_hard_results['overall']['auc']:.3f}")
                
                # 以困难集F1为模型选择标准
                if dev_hard_results['overall']['f1'] > best_f1:
                    best_f1 = dev_hard_results['overall']['f1']
                    best_model = name
            else:
                # 如果没有困难集，用标准集F1
                if dev_results['overall']['f1'] > best_f1:
                    best_f1 = dev_results['overall']['f1']
                    best_model = name
            
            # 保存模型
            model_file = self.output_dir / f"{name}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"  💾 模型保存: {model_file}")
            
            self.models[name] = model
            self.results[name] = {
                "dev": dev_results,
                "dev_hard": dev_hard_results if len(X_dev_hard) > 0 else None
            }
        
        # 保存结果
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print(f"🏆 最佳模型: {best_model} (困难集F1: {best_f1:.3f})")
        print("=" * 60)
        print(f"\n✅ 结果保存: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="S7 Phase 3: 模型训练")
    parser.add_argument("--features_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/06_features",
                        help="特征目录")
    parser.add_argument("--output_dir", type=str,
                        default="/home/aigc/aigc_checker/phase3/07_models",
                        help="输出目录")
    
    args = parser.parse_args()
    
    trainer = ModelTrainerPhase3(args.features_dir, args.output_dir)
    trainer.run()


if __name__ == "__main__":
    main()
