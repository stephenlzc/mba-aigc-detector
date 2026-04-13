#!/usr/bin/env python3
"""
MBA论文AIGC检测 - 推理脚本
"""

import argparse
import json
from pathlib import Path

from fusion_detector_cnki_calibrated import create_calibrated_detector
from feature_extractor import FeatureExtractor
from document_processor import DocumentProcessor


def detect_document(input_path: Path, output_path: Path = None, models_dir: str = None):
    """
    检测单个文档
    
    Args:
        input_path: 输入文档路径
        output_path: 输出结果路径（可选）
        models_dir: 模型目录路径（可选，默认从环境变量MBA_AIGC_MODEL_DIR获取）
    """
    import os
    
    print("="*60)
    print("MBA论文AIGC检测")
    print("="*60)
    print(f"\n输入文件: {input_path}")
    
    # 获取模型目录
    if models_dir is None:
        models_dir = os.getenv("MBA_AIGC_MODEL_DIR", "./models")
    
    print(f"模型目录: {models_dir}")
    
    # 初始化
    print("\n初始化模型...")
    detector = create_calibrated_detector(models_dir=models_dir)
    extractor = FeatureExtractor()
    processor = DocumentProcessor()
    
    # 处理文档
    print("读取文档...")
    paragraphs = processor.process(input_path)
    print(f"  检测到 {len(paragraphs)} 个段落")
    
    # 提取特征
    print("提取特征...")
    features = []
    for i, para in enumerate(paragraphs):
        feat = extractor.extract(para)
        features.append(feat)
        if (i + 1) % 10 == 0:
            print(f"  已处理: {i+1}/{len(paragraphs)}")
    
    # 检测
    print("\n进行检测...")
    result = detector.predict_document(features)
    
    # 格式化输出
    print("\n" + detector.format_result(result))
    
    # 保存结果
    if output_path:
        import numpy as np
        
        # 自定义JSON编码器处理numpy类型
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        output = {
            "input_file": str(input_path),
            "doc_calibrated_score": float(result['doc_calibrated_score']),
            "doc_raw_score": float(result['doc_raw_score']),
            "risk_level": result['risk_level'],
            "ai_paragraph_count": int(result['ai_paragraph_count']),
            "total_paragraphs": int(result['total_paragraphs']),
            "ai_ratio": float(result['ai_ratio']),
            "paragraph_results": result['paragraph_results'][:30]  # 保存前30段
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"\n✓ JSON结果已保存: {output_path}")
        
        # 自动生成HTML报告(全文标注版)
        try:
            from report_generator_html import HTMLReportGenerator
            html_path = output_path.with_suffix('.html')
            generator = HTMLReportGenerator(output, str(input_path), paragraphs)
            generator.generate(str(html_path))
        except Exception as e:
            print(f"  HTML报告生成失败: {e}")


def main():
    import os
    
    parser = argparse.ArgumentParser(description="MBA论文AIGC检测")
    parser.add_argument("input", type=str, help="输入文件路径 (pdf/docx/txt)")
    parser.add_argument("--output", "-o", type=str, help="输出JSON文件路径")
    parser.add_argument("--models", "-m", type=str, 
                       default=os.getenv("MBA_AIGC_MODEL_DIR", "./models"),
                       help="模型目录路径 (默认: ./models 或环境变量MBA_AIGC_MODEL_DIR)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    models_dir = args.models
    
    if not input_path.exists():
        print(f"错误: 文件不存在 - {input_path}")
        return
    
    detect_document(input_path, output_path, models_dir)


if __name__ == "__main__":
    main()
