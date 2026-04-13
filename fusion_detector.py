#!/usr/bin/env python3
"""
5模型并联融合检测器
核心逻辑: 任一模型阳性 → 整体阳性（OR逻辑）
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FusionResult:
    """融合检测结果"""
    is_ai: bool
    confidence: float
    triggered_by: str
    all_model_probs: Dict[str, float]
    threshold: float
    mode: str
    inference_time_ms: float = 0.0


class FusionAIGCDetector:
    """
    5模型并联敏感检测器
    
    核心特性:
    - 5个异构模型并联运行
    - OR逻辑: 任一模型阳性 → 整体阳性
    - 阈值可调: 0.30(高敏感) / 0.35(平衡) / 0.45(严格)
    - 目标: Recall>=0.85, 假阴性率<5%
    """
    
    # 预定义模式
    MODES = {
        "sensitive": {"threshold": 0.30, "expected_recall": 0.90, "expected_precision": 0.70},
        "balanced": {"threshold": 0.35, "expected_recall": 0.85, "expected_precision": 0.75},
        "strict": {"threshold": 0.45, "expected_recall": 0.75, "expected_precision": 0.85},
    }
    
    def __init__(self, models_dir: Union[str, Path], mode: str = "balanced", 
                 threshold: Optional[float] = None):
        """
        初始化融合检测器
        
        Args:
            models_dir: 基础模型目录
            mode: 检测模式 (sensitive/balanced/strict)
            threshold: 自定义阈值（覆盖mode配置）
        """
        self.models_dir = Path(models_dir)
        self.mode = mode
        
        # 设置阈值
        if threshold is not None:
            self.threshold = threshold
        elif mode in self.MODES:
            self.threshold = self.MODES[mode]["threshold"]
        else:
            self.threshold = 0.35
        
        # 加载5个基础模型
        self.models = self._load_models()
        
        print(f"✓ FusionDetector初始化完成")
        print(f"  模式: {mode}, 阈值: {self.threshold}")
        print(f"  已加载模型: {list(self.models.keys())}")
    
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
            model_path = self.models_dir / f"{name}_model.pkl"
            with open(model_path, 'rb') as f:
                model_data['model'] = pickle.load(f)
            
            # 加载selector（如果存在）
            selector_path = self.models_dir / f"{name}_selector.pkl"
            if selector_path.exists():
                with open(selector_path, 'rb') as f:
                    model_data['selector'] = pickle.load(f)
            else:
                model_data['selector'] = None
            
            # 加载元信息
            meta_path = self.models_dir / f"{name}_meta.json"
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                model_data['use_bert'] = meta.get('use_bert', False)
                model_data['feature_dim'] = meta.get('feature_dim', 0)
            
            models[name] = model_data
        
        return models
    
    def set_mode(self, mode: str):
        """动态切换模式"""
        if mode not in self.MODES:
            raise ValueError(f"未知模式: {mode}. 可用模式: {list(self.MODES.keys())}")
        
        self.mode = mode
        self.threshold = self.MODES[mode]["threshold"]
        print(f"✓ 模式切换为: {mode}, 阈值: {self.threshold}")
    
    def set_threshold(self, threshold: float):
        """自定义阈值（高级用户）"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("阈值必须在0.0-1.0之间")
        
        self.threshold = threshold
        self.mode = f"custom_{threshold:.2f}"
        print(f"✓ 阈值设置为: {threshold}")
    
    def predict(self, combined_features: Optional[np.ndarray] = None,
                bert_features: Optional[np.ndarray] = None) -> FusionResult:
        """
        融合预测
        
        Args:
            combined_features: 781维组合特征 (13统计+768BERT)
            bert_features: 768维BERT特征
        
        Returns:
            FusionResult: 融合检测结果
        """
        import time
        start_time = time.time()
        
        all_probs = {}
        
        # 遍历5个模型进行预测
        for name, model_data in self.models.items():
            # 准备输入特征
            if model_data['use_bert']:
                if bert_features is None and combined_features is not None:
                    # 从combined_features中提取BERT特征
                    X_input = combined_features[:, -768:]
                else:
                    X_input = bert_features
            else:
                if model_data['selector'] is not None:
                    X_input = combined_features
                else:
                    X_input = combined_features
            
            # 应用selector
            if model_data['selector'] is not None:
                X_input = model_data['selector'].transform(X_input)
            
            # 预测概率
            prob = model_data['model'].predict_proba(X_input)[0, 1]
            all_probs[name] = prob
            
            # OR逻辑: 任一模型超过阈值即返回阳性（早停优化）
            if prob > self.threshold:
                inference_time = (time.time() - start_time) * 1000
                return FusionResult(
                    is_ai=True,
                    confidence=float(prob),
                    triggered_by=name,
                    all_model_probs=all_probs.copy(),
                    threshold=self.threshold,
                    mode=self.mode,
                    inference_time_ms=inference_time
                )
        
        # 所有模型都未触发，返回最高概率的模型结果
        max_model = max(all_probs, key=all_probs.get)
        max_prob = all_probs[max_model]
        
        inference_time = (time.time() - start_time) * 1000
        
        return FusionResult(
            is_ai=max_prob > self.threshold,
            confidence=float(max_prob),
            triggered_by=max_model if max_prob > self.threshold else "none",
            all_model_probs=all_probs,
            threshold=self.threshold,
            mode=self.mode,
            inference_time_ms=inference_time
        )
    
    def predict_proba(self, combined_features: Optional[np.ndarray] = None,
                      bert_features: Optional[np.ndarray] = None) -> float:
        """
        返回融合概率（所有模型概率的最大值）
        
        注意: 这是并联系统的概率，不是平均概率
        """
        result = self.predict(combined_features, bert_features)
        return result.confidence
    
    def get_model_contributions(self) -> Dict[str, Dict]:
        """获取各模型在测试集上的贡献统计"""
        return {
            name: {
                "feature_dim": data['feature_dim'],
                "use_bert": data['use_bert'],
                "has_selector": data['selector'] is not None
            }
            for name, data in self.models.items()
        }
    
    def explain_prediction(self, result: FusionResult) -> str:
        """
        生成预测解释
        """
        lines = [
            f"AIGC检测结果: {'高风险' if result.is_ai else '低风险'}",
            f"触发阈值: {result.threshold}",
            f"最高置信度: {result.confidence:.3f}",
            f"触发模型: {result.triggered_by}",
            "",
            "各模型概率:",
        ]
        
        for name, prob in sorted(result.all_model_probs.items(), 
                                  key=lambda x: x[1], reverse=True):
            marker = " ✓" if prob > result.threshold else ""
            lines.append(f"  {name:20s}: {prob:.3f}{marker}")
        
        return "\n".join(lines)


class DocumentFusionDetector:
    """文档级融合检测器"""
    
    def __init__(self, detector: FusionAIGCDetector):
        self.detector = detector
    
    def detect_paragraphs(self, paragraphs_features: List[Dict]) -> Dict:
        """
        检测多个段落
        
        Args:
            paragraphs_features: 每个段落的特征字典列表
                [{"combined_features": [...], "bert_features": [...]}, ...]
        
        Returns:
            检测结果汇总
        """
        paragraph_results = []
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        
        for i, features in enumerate(paragraphs_features):
            combined = np.array(features.get("combined_features", [])).reshape(1, -1)
            bert = np.array(features.get("bert_features", [])).reshape(1, -1)
            
            result = self.detector.predict(combined, bert)
            
            # 风险分级
            if result.confidence > 0.5:
                risk_level = "high"
                risk_counts["high"] += 1
            elif result.confidence > 0.35:
                risk_level = "medium"
                risk_counts["medium"] += 1
            else:
                risk_level = "low"
                risk_counts["low"] += 1
            
            paragraph_results.append({
                "paragraph_id": i,
                "is_ai": result.is_ai,
                "confidence": result.confidence,
                "triggered_by": result.triggered_by,
                "risk_level": risk_level,
                "all_probs": result.all_model_probs
            })
        
        # 计算文档级分数（加权平均）
        total = len(paragraph_results)
        if total > 0:
            doc_score = sum(r["confidence"] for r in paragraph_results) / total
        else:
            doc_score = 0.0
        
        return {
            "doc_ai_score": doc_score,
            "paragraphs": paragraph_results,
            "high_risk_count": risk_counts["high"],
            "medium_risk_count": risk_counts["medium"],
            "low_risk_count": risk_counts["low"],
            "risk_distribution": {
                "high": risk_counts["high"] / total if total > 0 else 0,
                "medium": risk_counts["medium"] / total if total > 0 else 0,
                "low": risk_counts["low"] / total if total > 0 else 0,
            }
        }


# 便捷函数
def create_fusion_detector(mode: str = "balanced") -> FusionAIGCDetector:
    """创建融合检测器的便捷函数"""
    base_dir = Path(__file__).parent / "base"
    return FusionAIGCDetector(models_dir=base_dir, mode=mode)


if __name__ == "__main__":
    # 简单测试
    print("="*60)
    print("FusionAIGCDetector 测试")
    print("="*60)
    
    detector = create_fusion_detector(mode="balanced")
    
    print("\n模式信息:")
    for mode, config in detector.MODES.items():
        print(f"  {mode:10s}: threshold={config['threshold']}, "
              f"recall>={config['expected_recall']}")
