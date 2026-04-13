---
language:
  - zh
  - en
tags:
  - aigc-detection
  - text-classification
  - decision-tree
  - chinese
  - academic-paper
  - mba-thesis
license: mit
library_name: scikit-learn
base_model:
  - hfl/chinese-roberta-wwm-ext
---

# MBA论文AIGC检测系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-orange.svg)](https://huggingface.co/stephenlzc/mba-aigc-detector)

基于5模型并联融合的MBA论文AIGC风险识别系统。

---

## 项目链接

| 平台 | 链接 | 说明 |
|------|------|------|
| **GitHub** | https://github.com/stephenlzc/mba-aigc-detector | 本仓库：完整代码、文档 |
| **HuggingFace** | https://huggingface.co/stephenlzc/mba-aigc-detector | 预训练模型权重（~60KB） |

## 特点

- 🎯 **高Recall设计**: 宁可假阳不要假阴，最大化检出率
- 🔗 **5模型并联**: OR逻辑，任一阳性即阳性
- 📊 **CNKI校准**: 输出分数接近知网AIGC检测AI特征值
- 🚀 **本地运行**: 支持ollama本地部署，保护隐私

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载模型

从HuggingFace下载预训练模型（总共~60KB，轻量高效）：

```bash
# 方法1: 使用huggingface-cli
huggingface-cli download stephenlzc/mba-aigc-detector --local-dir ./models

# 方法2: 使用git-lfs
git lfs install
git clone https://huggingface.co/stephenlzc/mba-aigc-detector ./models

# 方法3: 直接下载
wget https://huggingface.co/stephenlzc/mba-aigc-detector/resolve/main/select5_tree_d2.joblib -O ./models/select5_tree_d2.joblib
wget https://huggingface.co/stephenlzc/mba-aigc-detector/resolve/main/select10_tree_d2.joblib -O ./models/select10_tree_d2.joblib
wget https://huggingface.co/stephenlzc/mba-aigc-detector/resolve/main/select15_tree_d3.joblib -O ./models/select15_tree_d3.joblib
wget https://huggingface.co/stephenlzc/mba-aigc-detector/resolve/main/select20_tree_d2.joblib -O ./models/select20_tree_d2.joblib
wget https://huggingface.co/stephenlzc/mba-aigc-detector/resolve/main/bert_tree_d1.joblib -O ./models/bert_tree_d1.joblib
```

### 单文档检测

```python
from fusion_detector import FusionDetector
from feature_extractor import FeatureExtractor
from document_processor import DocumentProcessor

# 初始化
detector = FusionDetector(models_dir="./models")
extractor = FeatureExtractor()
processor = DocumentProcessor()

# 处理文档
text = processor.read_pdf("paper.pdf")
paragraphs = processor.split_paragraphs(text)

# 提取特征并检测
features = [extractor.extract(p) for p in paragraphs]
result = detector.predict_document(features)

print(f"AIGC分数: {result['doc_calibrated_score']:.1%}")
print(f"风险等级: {result['risk_level']}")
```

### 使用校准版检测器（推荐）

```python
from fusion_detector_cnki_calibrated import create_calibrated_detector

# 方法1: 通过参数指定模型目录
detector = create_calibrated_detector(models_dir="./models")

# 方法2: 通过环境变量设置模型目录
import os
os.environ["MBA_AIGC_MODEL_DIR"] = "./models"
detector = create_calibrated_detector()

result = detector.predict_document(features)

# 输出接近CNKI的AI特征值
print(f"CNKI校准AIGC分数: {result['doc_calibrated_score']:.1%}")
```

### 命令行使用

```bash
# 基本使用
python inference.py paper.pdf

# 指定输出文件
python inference.py paper.pdf -o result.json

# 指定模型目录
python inference.py paper.pdf --models ./models -o result.json

# 或通过环境变量设置模型目录
export MBA_AIGC_MODEL_DIR="./models"
python inference.py paper.pdf
```

## 模型架构

```
输入文本
    │
    ├─→ Select5_Tree ──┐
    ├─→ Select10_Tree ─┤
    ├─→ Select15_Tree ─┼─→ OR Gate ─→ CNKI校准 ─→ 输出
    ├─→ Select20_Tree ─┤  (任一阳性即阳性)
    └─→ BERT_Tree ─────┘
```

### 预训练模型

本系统的BERT特征提取基于中文RoBERTa模型：

| 模型 | 来源 | 用途 | 链接 |
|------|------|------|------|
| **chinese-roberta-wwm-ext** | 哈工大讯飞联合实验室 | 768维语义特征提取 | https://huggingface.co/hfl/chinese-roberta-wwm-ext |

RoBERTa (Robustly optimized BERT approach) 采用全词掩码(Whole Word Masking)技术，对中文文本有更好的表征能力。

### 模型大小优势

本系统采用决策树架构，模型文件极小：

| 模型 | 文件大小 | 说明 |
|------|----------|------|
| select5_tree_d2 | ~4KB | 5特征决策树 |
| select10_tree_d2 | ~8KB | 10特征决策树 |
| select15_tree_d3 | ~16KB | 15特征决策树 |
| select20_tree_d2 | ~12KB | 20特征决策树 |
| bert_tree_d1 | ~6KB | BERT特征决策树 |
| **总计** | **~60KB** | 全部5个模型 |

相比神经网络模型（通常几MB到几GB），本系统模型体积**缩小了99%以上**，适合轻量部署和资源受限环境。

## 性能指标

| 指标 | Train | Test |
|------|-------|------|
| Recall | 0.895 | 0.974 |
| 假阴性率 | 10.5% | 2.6% |

## 目录结构

```
.
├── fusion_detector.py              # 基础检测器
├── fusion_detector_cnki_calibrated.py  # CNKI校准版 ⭐推荐
├── feature_extractor.py            # 特征提取
├── document_processor.py           # 文档处理
├── inference.py                    # 推理脚本
├── requirements.txt                # 依赖
└── README.md                       # 本文件
```

## 配置文件

### 校准配置 (calibration_config.json)

```json
{
  "calibration_factor": 0.3,
  "thresholds": {
    "select5_tree_d2": 0.50,
    "select10_tree_d2": 0.50,
    "select15_tree_d3": 0.50,
    "select20_tree_d2": 0.50,
    "bert_tree_d1": 0.40
  },
  "risk_thresholds": {
    "low": 0.15,
    "medium": 0.30,
    "high": 0.50
  }
}
```

## 风险等级说明

| 等级 | AIGC分数 | 说明 |
|------|----------|------|
| 低风险 | <15% | 疑似人工写作 |
| 中风险 | 15-30% | 部分AI痕迹 |
| 较高风险 | 30-50% | 明显AI痕迹 |
| 高风险 | >50% | 高度疑似AI生成 |

## 免责声明

本系统输出仅供参考，不构成对学术不端行为的直接认定，也不替代学校官方检测系统或CNKI等权威检测结果。

## License

MIT License - 详见 [LICENSE](LICENSE) 文件
