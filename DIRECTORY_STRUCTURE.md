# MBA-AIGC-Detector 目录结构

## 核心脚本 (scripts/)

### 数据准备阶段
- `S1_5_quality_control_phase3.py` - S1.5: 数据清洗与质量控制
- `S2_paragraph_segmentation_phase3.py` - S2: 段落切分
- `S3_metadata_extraction_phase3.py` - S3: 元数据提取
- `S4_dataset_split_phase3.py` - S4: 时间切分

### AI变体生成阶段 ⭐
- `S5_concurrent.py` - S5: 并发AI变体生成
  - 3槽位并发架构
  - 15分钟轮换机制
  - 9分钟Fallback
  - 6种变体类型

### 特征工程阶段 ⭐
- `S6_feature_engineering_phase3.py` - S6: BERT+统计特征提取
  - BERT特征: 768维
  - 统计特征: 13维
  - 总计: 781维

### 模型训练阶段 ⭐
- `S7_train_models_phase3.py` - S7: 双轨模型训练
  - GBDT/RF/LR三种模型
  - 自适应特征加载

### 评估阶段 ⭐
- `S8_evaluate_and_inference_phase3.py` - S8: 双轨评估
  - 标准集+困难集评估
  - 过拟合检测

### 辅助脚本
- `fix_year_and_split.py` - 字段修复工具

## 文档

| 文档 | 内容 |
|------|------|
| `README.md` | 项目首页，快速开始指南 |
| `REPORT.md` | 完整方法论报告（含执行结果、局限性声明） |
| `DIRECTORY_STRUCTURE.md` | 本文件，目录结构说明 |
| `PRE_PUSH_CHECKLIST.md` | 推送前检查清单 |

## 配置文件

- `.env.example` - 环境变量模板（API Keys等）
- `.gitignore` - Git忽略规则

## 使用说明

### 1. 环境准备
```bash
pip install numpy scikit-learn transformers torch httpx python-dotenv
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 填入真实的API Keys
```

### 3. 执行完整流程
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

### 4. 补充数据（如需要）
```bash
python scripts/S5_concurrent.py \
  --supplement_mode \
  --supplement_targets '[
    {"split": "test", "variant": "hard_human", "target": 100}
  ]'

python scripts/fix_year_and_split.py
# 重新执行 S6-S8
```

## 性能指标

- **F1**: 0.967
- **AUC**: 0.994
- **Precision**: 0.997
- **Recall**: 0.858

详见 REPORT.md 完整文档。

## 重要提示

⚠️ **局限性声明**: 本系统存在学科局限、时间局限、语言局限等，详见 REPORT.md 第5章。

⚠️ **免责声明**: 本系统仅供参考，不作为学术诚信认定的唯一依据。
