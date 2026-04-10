# MBA-AIGC-Detector

> 🔍 基于BERT的MBA论文AIGC风险检测系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![transformers](https://img.shields.io/badge/transformers-4.30+-yellow.svg)](https://huggingface.co/transformers/)

---

## 🌟 项目亮点

- **高精度检测**: F1 = **0.967**, AUC = **0.994**
- **多维度变体**: 6种AI生成变体类型
- **双轨评估**: 标准集 + 困难集双重验证
- **工程化流程**: S1-S8完整流水线
- **可解释性**: BERT特征 + 统计特征融合

---

## 📊 性能指标

| 指标 | 要求 | 实际值 | 状态 |
|------|------|--------|------|
| Precision | ≥0.75 | **0.997** | ✅ 超标33% |
| Recall | ≥0.70 | **0.858** | ✅ 超标23% |
| F1 | ≥0.72 | **0.967** | ✅ 超标34% |
| PR-AUC | ≥0.78 | **0.994** | ✅ 超标27% |

---

## 🗂️ 项目结构

```
mba-aigc-detector/
├── scripts/                    # 核心脚本
│   ├── S1_5_quality_control_phase3.py      # S1.5: 数据清洗
│   ├── S2_paragraph_segmentation_phase3.py # S2: 段落切分
│   ├── S3_metadata_extraction_phase3.py    # S3: 元数据提取
│   ├── S4_dataset_split_phase3.py          # S4: 时间切分
│   ├── S5_concurrent.py                    # S5: AI变体生成 ⭐
│   ├── S6_feature_engineering_phase3.py    # S6: 特征工程 ⭐
│   ├── S7_train_models_phase3.py           # S7: 模型训练 ⭐
│   └── S8_evaluate_and_inference_phase3.py # S8: 评估测试 ⭐
├── REPORT.md                   # 完整方法论报告
├── DIRECTORY_STRUCTURE.md      # 目录结构说明
└── README.md                   # 本文件
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐，用于GPU加速)
- 16GB+ RAM
- 50GB+ 磁盘空间

### 安装依赖

```bash
pip install numpy scikit-learn transformers torch httpx python-dotenv
```

### 配置环境变量

创建 `.env` 文件：

```bash
# API Keys (用于S5 AI变体生成)
SERVERCHAN_SENDKEY=your_sendkey
DEEPSEEK_API_KEY=your_deepseek_key
ALIYUN_API_KEY=your_aliyun_key
MOONSHOT_API_KEY=your_moonshot_key

# 可选: ZHIPU_API_KEY=your_zhipu_key
```

### 执行流程

```bash
# 1. 数据准备 (S1-S4)
python scripts/S1_5_quality_control_phase3.py
python scripts/S2_paragraph_segmentation_phase3.py
python scripts/S3_metadata_extraction_phase3.py
python scripts/S4_dataset_split_phase3.py

# 2. AI变体生成 (S5)
python scripts/S5_concurrent.py

# 3. 特征工程 (S6)
python scripts/S6_feature_engineering_phase3.py \
  --model_path /path/to/bert_model

# 4. 模型训练 (S7)
python scripts/S7_train_models_phase3.py

# 5. 评估测试 (S8)
python scripts/S8_evaluate_and_inference_phase3.py
```

---

## 📖 核心特性

### S5: 并发AI变体生成

- **3槽位并发架构**: train/dev/test并行处理
- **15分钟轮换机制**: 避免API限流
- **9分钟Fallback**: 自动切换备用模型
- **6种变体类型**:
  - `hard_human`: 轻微润色
  - `multi_round_rewrite`: 多轮改写
  - `partial_rewrite`: 部分改写
  - `back_translation_mix`: 回译混合
  - `polish_manual_mix`: 手动润色混合
  - `standard_rewrite`: 标准改写

### S6: 特征工程

- **BERT特征**: 768维 (CLS token)
- **统计特征**: 13维
  - 字符数、句子数、中英文比例
  - 标点统计、词汇多样性
  - 引用/公式/表格/图片标记
- **总计**: 781维特征

### S7: 模型训练

- **双轨训练**: 标准样本 + 困难样本
- **3种模型**: GBDT, Random Forest, Logistic Regression
- **自适应加载**: 自动检测特征格式
- **最佳模型选择**: 基于困难集F1

### S8: 双轨评估

- **标准集评估**: 常规测试集
- **困难集评估**: hard_human变体测试
- **过拟合检测**: 自动风险预警
- **不确定样本**: 概率0.4-0.6样本统计

---

## 📚 详细文档

- [完整方法论报告](./REPORT.md) - 详细的使用指南和优化说明
- [目录结构说明](./DIRECTORY_STRUCTURE.md) - 项目结构详解

---

## 🔧 高级用法

### 补充特定变体

```bash
python scripts/S5_concurrent.py \
  --supplement_mode \
  --supplement_targets '[
    {"split": "test", "variant": "hard_human", "target": 100}
  ]'
```

### 修复year字段

```bash
python scripts/fix_year_and_split.py
```

---

## 📈 数据规模

| 类型 | 数量 |
|------|------|
| 人类样本 | 21,876 条 |
| AI变体 | 5,958 条 |
| **总计** | **27,834 条** |

### 6种变体分布

| 变体类型 | 数量 |
|---------|------|
| hard_human | 1,457 条 |
| standard_rewrite | 1,162 条 |
| polish_manual_mix | 1,035 条 |
| partial_rewrite | 921 条 |
| back_translation_mix | 753 条 |
| multi_round_rewrite | 630 条 |

---

## 🎯 应用场景

- 📄 MBA论文AIGC风险检测
- 📊 段落级概率输出
- 🔍 可解释检测结果
- 📋 结构化检测报告

---

## 📝 引用

如果你使用了本项目，请引用：

```bibtex
@software{mba_aigc_detector,
  title = {MBA-AIGC-Detector: 基于BERT的MBA论文AIGC风险检测系统},
  author = {stephenlzc},
  year = {2026},
  url = {https://github.com/stephenlzc/mba-aigc-detector}
}
```

---

## 📄 License

MIT License

---

## 🤝 贡献

欢迎提交Issue和PR！

---

**Made with ❤️ for MBA Academic Integrity**
