# MBA-AIGC-Detector: 完整方法论报告

> 🔍 基于BERT的MBA论文AIGC风险检测系统 - Phase 3 完整方法论

[![Performance](https://img.shields.io/badge/F1-0.967-brightgreen.svg)]()
[![AUC](https://img.shields.io/badge/AUC-0.994-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

---

## 📋 目录

1. [执行摘要](#一执行摘要)
2. [项目概述](#二项目概述)
3. [完整脚本清单](#三完整脚本清单)
4. [执行结果](#四执行结果)
5. [局限性声明](#五局限性声明-limitation)
6. [复用指南](#六复用指南)
7. [最佳实践](#七最佳实践)

---

## 一、执行摘要

### 1.1 核心成果
| 指标 | 要求 | 实际值 | 状态 |
|------|------|--------|------|
| **Precision** | ≥0.75 | **0.997** | ✅ 超标33% |
| **Recall** | ≥0.70 | **0.858** | ✅ 超标23% |
| **F1** | ≥0.72 | **0.967** | ✅ 超标34% |
| **PR-AUC** | ≥0.78 | **0.994** | ✅ 超标27% |

### 1.2 执行时间
| 阶段 | 耗时 |
|------|------|
| S1-S4 数据准备 | 视数据量 |
| S5 AI变体生成 | 数小时 (API调用) |
| S6 特征工程 | ~5分钟 |
| S7 模型训练 | ~8分钟 |
| S8 评估测试 | <1分钟 |

### 1.3 数据规模
- **总样本**: 27,834 条
- **人类样本**: 21,876 条
- **AI变体**: 5,958 条
- **变体类型**: 6 种

---

## 二、项目概述

### 2.1 核心目标
- 支持上传 `docx/pdf/txt/md` 格式的MBA论文
- 输出段落级AIGC风险概率
- 提供可解释的检测依据
- 生成结构化检测报告

### 2.2 与旧结果对比
| 模型 | 旧F1 | 新F1 | 提升幅度 |
|------|------|------|---------|
| **GBDT** | 0.696 | **0.967** | ⬆️ **+38.9%** |
| RF | 0.662 | 0.945 | ⬆️ +42.7% |
| LR | 0.545 | 0.964 | ⬆️ +76.8% |

---

## 三、完整脚本清单

### 3.1 数据准备阶段 (S1-S4)

#### S1_5_quality_control_phase3.py
**功能**: 数据清洗与质量控制

**核心逻辑**:
```python
# 清洗规则
- 移除封面、版权页、目录页
- 保留正文核心内容
- 识别章节角色(chapter_role)
```

**使用**:
```bash
python scripts/S1_5_quality_control_phase3.py \
  --input_dir 01_markdown_output \
  --output_dir 01_qc_output
```

---

#### S2_paragraph_segmentation_phase3.py
**功能**: 段落切分

**核心参数**:
```python
MIN_PARAGRAPH_LENGTH = 50
MAX_PARAGRAPH_LENGTH = 2000
```

**输出字段**: text, doc_id, para_index, para_type, chapter_role

---

#### S3_metadata_extraction_phase3.py
**功能**: 元数据提取

**提取字段**: year, school, major, author, title, char_count, para_count

---

#### S4_dataset_split_phase3.py
**功能**: 时间切分

**切分规则**:
```python
TRAIN_YEARS = [2015-2020]  # 2015-2020
dev_YEARS   = [2021-2022]  # 2021-2022
TEST_YEARS  = [2023-2024]  # 2023-2024
```

---

### 3.2 AI变体生成阶段 (S5) ⭐核心

#### S5_concurrent.py
**功能**: 并发AI变体生成

**核心优化**:
1. **3槽位并发**: train/dev/test并行
2. **15分钟轮换**: 避免API限流
3. **9分钟Fallback**: 自动切换备用模型
4. **补充模式**: 精确补充缺口数据

**6种变体类型**:
- `hard_human`: 轻微润色，保留人类痕迹
- `multi_round_rewrite`: 多轮改写，平衡AI和人类风格
- `partial_rewrite`: 部分改写
- `back_translation_mix`: 回译混合
- `polish_manual_mix`: 手动润色混合
- `standard_rewrite`: 标准改写

**使用**:
```bash
# 正常模式
python scripts/S5_concurrent.py

# 补充模式
python scripts/S5_concurrent.py \
  --supplement_mode \
  --supplement_targets '[
    {"split": "test", "variant": "hard_human", "target": 100}
  ]'
```

---

### 3.3 特征工程阶段 (S6) ⭐核心

#### S6_feature_engineering_phase3.py
**功能**: BERT + 统计特征提取

**特征维度**:
- **BERT特征**: 768维 (CLS token)
- **统计特征**: 13维
  - 字符数、句子数、中英文比例
  - 标点统计、词汇多样性
  - 引用/公式/表格标记
- **总计**: 781维

**GPU优化**:
```python
BERT_BATCH_SIZE = 32  # GPU
CPU_BATCH_SIZE = 8    # CPU
```

**使用**:
```bash
python scripts/S6_feature_engineering_phase3.py \
  --model_path /path/to/bert_model \
  --split_dir 04_dataset_split \
  --variants_dir 05_ai_variants \
  --output_dir 06_features
```

---

### 3.4 模型训练阶段 (S7) ⭐核心

#### S7_train_models_phase3.py
**功能**: 双轨模型训练

**模型配置**:
```python
{
  "gbdt": GradientBoostingClassifier(n_estimators=100, max_depth=5),
  "rf": RandomForestClassifier(n_estimators=100, max_depth=10),
  "lr": LogisticRegression(max_iter=1000)
}
```

**核心优化**:
- 自适应特征加载
- 双轨训练（标准+困难样本）
- 困难集F1作为模型选择标准

**使用**:
```bash
python scripts/S7_train_models_phase3.py \
  --features_dir 06_features \
  --output_dir 07_models
```

---

### 3.5 评估阶段 (S8) ⭐核心

#### S8_evaluate_and_inference_phase3.py
**功能**: 双轨评估

**评估指标**:
- Precision/Recall/F1
- AUC
- 混淆矩阵
- 不确定样本比例

**核心功能**:
- 过拟合检测
- 自适应特征加载

**使用**:
```bash
python scripts/S8_evaluate_and_inference_phase3.py \
  --features_dir 06_features \
  --models_dir 07_models \
  --output_dir 08_evaluation
```

---

### 3.6 辅助脚本

#### fix_year_and_split.py
**功能**: 修复year字段，检查split分配

**使用**:
```bash
python scripts/fix_year_and_split.py
```

---

## 四、执行结果

### 4.1 模型性能

#### Test集结果
| 模型 | F1 | AUC | Precision | Recall |
|------|-----|-----|-----------|--------|
| **GBDT** | **0.967** | **0.994** | 0.997 | 0.858 |
| RF | 0.945 | 0.996 | 1.000 | 0.735 |
| LR | 0.964 | 0.994 | 0.992 | 0.844 |

#### Test Hard集结果
| 模型 | F1 | AUC | Precision | Recall |
|------|-----|-----|-----------|--------|
| **GBDT** | **0.941** | **0.993** | 0.997 | 0.939 |
| RF | 0.909 | 0.997 | 1.000 | 0.839 |
| LR | 0.919 | 0.996 | 0.988 | 0.911 |

### 4.2 变体类型分布
| 变体类型 | 数量 |
|---------|------|
| hard_human | 1,457 条 |
| standard_rewrite | 1,162 条 |
| polish_manual_mix | 1,035 条 |
| partial_rewrite | 921 条 |
| back_translation_mix | 753 条 |
| multi_round_rewrite | 630 条 |

### 4.3 关键改进
1. **数据补充**: S5补充5438条高质量AI变体
2. **字段修复**: 修复所有year=0问题
3. **结构统一**: 18个JSON文件结构一致
4. **双轨评估**: 标准集+困难集双重验证
5. **过拟合控制**: 所有模型均为低风险

---

## 五、局限性声明 (Limitation)

### ⚠️ 重要提示

本系统存在以下局限性，使用时请务必注意：

### 5.1 数据局限

| 局限 | 说明 | 影响 |
|------|------|------|
| **学科局限** | 仅针对MBA/管理类硕士论文训练 | 跨学科泛化能力未验证 |
| **时间局限** | 训练数据为2015-2024年论文 | 对最新AI模型生成文本敏感度可能下降 |
| **语言局限** | 仅支持中文文本检测 | 英文/多语言论文无法准确检测 |
| **格式局限** | 主要针对传统论文结构 | 非标准格式文档检测效果未知 |

### 5.2 技术局限

| 局限 | 说明 | 建议 |
|------|------|------|
| **概率性输出** | 输出为风险概率，非确定性判断 | 需结合人工审核，不可直接作为处罚依据 |
| **段落级检测** | 仅支持段落级，非整篇分析 | 需配合文档级聚合策略 |
| **阈值敏感** | 分类阈值影响结果显著 | 建议根据业务场景调优阈值 |
| **对抗样本** | 对刻意规避的改写文本检测能力有限 | 需持续更新变体类型 |

### 5.3 使用限制

- **非官方检测工具**: 本系统不替代学校官方查重/检测系统
- **仅作参考**: 检测结果仅供参考，不作为学术诚信认定的唯一依据
- **免责声明**: 使用本系统产生的任何后果由用户自行承担

### 5.4 维护声明

- 本系统需要持续维护，建议定期更新训练数据
- API服务可能变更，需关注第三方服务状态
- 模型可能随时间退化，建议定期重训练

---

## 六、复用指南

### 6.1 环境准备

```bash
# 安装依赖
pip install numpy scikit-learn transformers torch httpx python-dotenv

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入API Keys
```

### 6.2 完整执行流程

```bash
# S1-S4: 数据准备
python scripts/S1_5_quality_control_phase3.py
python scripts/S2_paragraph_segmentation_phase3.py
python scripts/S3_metadata_extraction_phase3.py
python scripts/S4_dataset_split_phase3.py

# S5: AI变体生成
python scripts/S5_concurrent.py

# S6-S8: 特征工程、训练、评估
python scripts/S6_feature_engineering_phase3.py
python scripts/S7_train_models_phase3.py
python scripts/S8_evaluate_and_inference_phase3.py
```

### 6.3 增量补充

当数据不足时：
```bash
python scripts/S5_concurrent.py \
  --supplement_mode \
  --supplement_targets '[
    {"split": "test", "variant": "hard_human", "target": 100}
  ]'

python scripts/fix_year_and_split.py
# 重新执行 S6-S8
```

---

## 七、最佳实践

1. **数据质量优先**: 确保S1-S4的数据质量，这是后续所有步骤的基础
2. **充分的数据**: 每种变体至少150条，3个split都要覆盖
3. **监控API状态**: S5阶段关注进度汇报，及时处理错误
4. **验证特征**: S6完成后检查特征维度是否为781
5. **选择最佳模型**: 以困难集F1为选择标准，而非标准集
6. **阈值调优**: 根据业务场景调整分类阈值
7. **持续监控**: 监控不确定样本比例，当前0.5%-1.6%
8. **定期更新**: 定期补充新变体，保持模型时效性

---

## 📚 补充文档

- [README.md](./README.md) - 项目快速开始指南
- [DIRECTORY_STRUCTURE.md](./DIRECTORY_STRUCTURE.md) - 目录结构说明
- [PRE_PUSH_CHECKLIST.md](./PRE_PUSH_CHECKLIST.md) - 推送前检查清单

---

**Made with ❤️ for MBA Academic Integrity**

**最后更新**: 2026-04-10
