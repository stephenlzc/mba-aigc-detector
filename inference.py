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


def detect_document(input_path: Path, output_path: Path = None):
    """
    检测单个文档
    
    Args:
        input_path: 输入文档路径
        output_path: 输出结果路径（可选）
    """
    print("="*60)
    print("MBA论文AIGC检测")
    print("="*60)
    print(f"\n输入文件: {input_path}")
    
    # 初始化
    print("\n初始化模型...")
    detector = create_calibrated_detector()
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
        output = {
            "input_file": str(input_path),
            "doc_calibrated_score": result['doc_calibrated_score'],
            "doc_raw_score": result['doc_raw_score'],
            "risk_level": result['risk_level'],
            "ai_paragraph_count": result['ai_paragraph_count'],
            "total_paragraphs": result['total_paragraphs'],
            "ai_ratio": result['ai_ratio'],
            "paragraph_results": result['paragraph_results'][:20]  # 只保存前20段
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 详细结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MBA论文AIGC检测")
    parser.add_argument("input", type=str, help="输入文件路径 (pdf/docx/txt)")
    parser.add_argument("--output", "-o", type=str, help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if not input_path.exists():
        print(f"错误: 文件不存在 - {input_path}")
        return
    
    detect_document(input_path, output_path)


if __name__ == "__main__":
    main()
