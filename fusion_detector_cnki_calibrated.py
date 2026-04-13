#!/usr/bin/env python3
"""
CNKI校准后的融合检测器
基于CNKI AIGC检测结果校准
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
from dataclasses import dataclass


@dataclass
class FusionResult:
    """融合检测结果"""
    is_ai: bool
    raw_probability: float
    calibrated_score: float  # 接近CNKI的AI特征值
    triggered_by: str
    all_model_probs: Dict[str, float]


class FusionDetectorCNKICalibrated:
    """
    CNKI校准后的5模型并联检测器
    
    校准因子: 0.3 (使输出接近CNKI的AI特征值)
    """
    
    # CNKI校准配置
    CALIBRATION_FACTOR = 0.3
    
    # 优化阈值（提高以降低敏感度）
    THRESHOLDS = {
        "select5_tree_d2": 0.50,
        "select10_tree_d2": 0.50,
        "select15_tree_d3": 0.50,
        "select20_tree_d2": 0.50,
        "bert_tree_d1": 0.40
    }
    
    # 风险等级阈值
    RISK_THRESHOLDS = {
        "low": 0.15,      # <15%: 低风险
        "medium": 0.30,   # 15-30%: 中风险
        "high": 0.50      # >30%: 高风险
    }
    
    def __init__(self, models_dir: Union[str, Path]):
        """
        初始化检测器
        
        Args:
            models_dir: 模型目录 (base_v2/)
        """
        self.models_dir = Path(models_dir)
        self.models = self._load_models()
        
        print(f"✓ CNKI校准检测器初始化完成")
        print(f"  校准因子: {self.CALIBRATION_FACTOR}")
        print(f"  模型: {list(self.models.keys())}")
    
    def _load_models(self) -> Dict:
        """加载5个基础模型"""
        models = {}
        model_names = [
            "select5_tree_d2",
            "select10_tree_d2", 
            "select15_tree_d3",
            "select20_tree_d2",
            "bert_tree_d1"
        ]
        
        for name in model_names:
            model_data = {}
            
            # 加载模型
            with open(self.models_dir / f"{name}_model.pkl", 'rb') as f:
                model_data['model'] = pickle.load(f)
            
            # 加载selector
            selector_path = self.models_dir / f"{name}_selector.pkl"
            if selector_path.exists():
                with open(selector_path, 'rb') as f:
                    model_data['selector'] = pickle.load(f)
            else:
                model_data['selector'] = None
            
            # 加载元信息
            with open(self.models_dir / f"{name}_meta.json", 'r') as f:
                meta = json.load(f)
                model_data['use_bert'] = meta.get('use_bert', False)
            
            models[name] = model_data
        
        return models
    
    def predict(self, combined_features: np.ndarray, bert_features: np.ndarray) -> FusionResult:
        """
        预测单段落
        
        Args:
            combined_features: 781维组合特征
            bert_features: 768维BERT特征
        
        Returns:
            FusionResult: 预测结果
        """
        all_probs = {}
        
        # 遍历5个模型
        for name, model_data in self.models.items():
            # 准备输入
            if model_data['use_bert']:
                X_input = bert_features.reshape(1, -1)
            else:
                X_input = combined_features.reshape(1, -1)
                if model_data['selector'] is not None:
                    X_input = model_data['selector'].transform(X_input)
            
            # 预测概率
            prob = model_data['model'].predict_proba(X_input)[0, 1]
            all_probs[name] = prob
            
            # OR逻辑: 任一模型超过阈值即阳性
            if prob > self.THRESHOLDS[name]:
                # CNKI校准
                calibrated = prob * self.CALIBRATION_FACTOR
                
                return FusionResult(
                    is_ai=True,
                    raw_probability=float(prob),
                    calibrated_score=float(calibrated),
                    triggered_by=name,
                    all_model_probs=all_probs.copy()
                )
        
        # 所有模型都未触发
        max_model = max(all_probs, key=all_probs.get)
        max_prob = all_probs[max_model]
        calibrated = max_prob * self.CALIBRATION_FACTOR
        
        return FusionResult(
            is_ai=max_prob > 0.50,  # 使用更高阈值
            raw_probability=float(max_prob),
            calibrated_score=float(calibrated),
            triggered_by=max_model,
            all_model_probs=all_probs
        )
    
    def predict_document(self, paragraph_features: List[Dict]) -> Dict:
        """
        预测整个文档
        
        Args:
            paragraph_features: 每个段落的特征列表
                [{"combined": [...], "bert": [...]}, ...]
        
        Returns:
            文档级结果
        """
        para_results = []
        total_calibrated_score = 0
        ai_para_count = 0
        
        for i, features in enumerate(paragraph_features):
            combined = np.array(features.get("combined_features", []))
            bert = np.array(features.get("bert_features", []))
            
            result = self.predict(combined, bert)
            
            para_results.append({
                "para_id": i,
                "is_ai": result.is_ai,
                "raw_probability": result.raw_probability,
                "calibrated_score": result.calibrated_score,
                "triggered_by": result.triggered_by
            })
            
            total_calibrated_score += result.calibrated_score
            if result.is_ai:
                ai_para_count += 1
        
        # 计算文档级分数（校准分数的平均值）
        doc_calibrated_score = total_calibrated_score / len(paragraph_features) if paragraph_features else 0
        
        # 确定风险等级
        if doc_calibrated_score < self.RISK_THRESHOLDS["low"]:
            risk_level = "低风险"
        elif doc_calibrated_score < self.RISK_THRESHOLDS["medium"]:
            risk_level = "中风险"
        elif doc_calibrated_score < self.RISK_THRESHOLDS["high"]:
            risk_level = "较高风险"
        else:
            risk_level = "高风险"
        
        return {
            "doc_calibrated_score": float(doc_calibrated_score),
            "doc_raw_score": float(total_calibrated_score / len(paragraph_features) / self.CALIBRATION_FACTOR) if paragraph_features else 0,
            "risk_level": risk_level,
            "ai_paragraph_count": ai_para_count,
            "total_paragraphs": len(paragraph_features),
            "ai_ratio": ai_para_count / len(paragraph_features) if paragraph_features else 0,
            "paragraph_results": para_results
        }
    
    def format_result(self, doc_result: Dict) -> str:
        """格式化输出结果"""
        lines = [
            "="*60,
            "AIGC检测结果 (CNKI校准版)",
            "="*60,
            f"",
            f"文档级AIGC分数: {doc_result['doc_calibrated_score']:.1%}",
            f"风险等级: {doc_result['risk_level']}",
            f"AI段落: {doc_result['ai_paragraph_count']}/{doc_result['total_paragraphs']} ({doc_result['ai_ratio']:.1%})",
            f"",
            f"说明: 该分数经过CNKI校准，更接近知网AIGC检测的AI特征值",
            f"",
            "风险阈值参考:",
            f"  - 低风险: <15%",
            f"  - 中风险: 15-30%",
            f"  - 较高风险: 30-50%",
            f"  - 高风险: >50%",
            "="*60
        ]
        
        return "\n".join(lines)


def create_calibrated_detector(models_dir: str = None) -> FusionDetectorCNKICalibrated:
    """
    创建校准检测器的便捷函数
    
    Args:
        models_dir: 模型目录路径，默认为当前目录下的models/
    """
    if models_dir is None:
        # 默认从环境变量或当前目录获取
        import os
        models_dir = os.getenv("MBA_AIGC_MODEL_DIR", "./models")
    
    base_dir = Path(models_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {base_dir}")
    
    return FusionDetectorCNKICalibrated(models_dir=base_dir)


if __name__ == "__main__":
    # 测试
    print("="*60)
    print("CNKI校准检测器测试")
    print("="*60)
    
    detector = create_calibrated_detector()
    
    print(f"\n校准因子: {detector.CALIBRATION_FACTOR}")
    print(f"阈值配置: {detector.THRESHOLDS}")
    print(f"风险阈值: {detector.RISK_THRESHOLDS}")
    
    print("\n✓ 检测器准备就绪")
    print("  使用方式: detector.predict_document(paragraph_features)")
