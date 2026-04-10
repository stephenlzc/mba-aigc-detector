#!/usr/bin/env python3
"""
修复变体文件中的year字段，并检查split分配是否正确
"""

import json
import os
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/home/aigc/aigc_checker/phase3")
VARIANTS_DIR = BASE_DIR / "05_ai_variants"
METADATA_FILE = BASE_DIR / "03_metadata" / "documents_metadata.json"
SPLIT_FILE = BASE_DIR / "04_dataset_split" / "split_index.json"

# 时间切分规则
TRAIN_YEARS = range(2015, 2021)  # 2015-2020
dev_YEARS = range(2021, 2023)    # 2021-2022
TEST_YEARS = range(2023, 2025)   # 2023-2024


def load_metadata():
    """加载元数据，建立doc_id到year的映射"""
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    doc_year_map = {}
    for info in data['metadata']:
        doc_id = info['doc_id']
        year = info.get('year', 0)
        doc_year_map[doc_id] = year
    
    return doc_year_map


def load_split_index():
    """加载split索引，获取正确的split分配"""
    with open(SPLIT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('doc_to_split', {})


def get_correct_split(year):
    """根据年份获取正确的split"""
    if year in TRAIN_YEARS:
        return 'train'
    elif year in dev_YEARS:
        return 'dev'
    elif year in TEST_YEARS:
        return 'test'
    else:
        return None


def process_variants_file(filepath, doc_year_map, split_index, variant_type):
    """处理单个变体文件"""
    filename = filepath.name
    parts = filename.replace('_variants.json', '').split('_')
    current_split = parts[0]  # train/dev/test
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计信息
    updated_count = 0
    year_0_count = 0
    wrong_split_items = []
    correct_items = []
    
    for item in data:
        doc_id = item.get('doc_id', '')
        
        # 获取正确的year
        correct_year = doc_year_map.get(doc_id, 0)
        
        # 更新year字段
        if item.get('year', 0) == 0 and correct_year > 0:
            item['year'] = correct_year
            updated_count += 1
        
        if item.get('year', 0) == 0:
            year_0_count += 1
        
        # 检查split是否正确
        correct_split = get_correct_split(item.get('year', 0))
        if correct_split and correct_split != current_split:
            # 标记为需要移动
            item['_target_split'] = correct_split
            wrong_split_items.append(item)
        else:
            correct_items.append(item)
    
    # 保存更新后的文件（只保留正确的items）
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(correct_items, f, ensure_ascii=False, indent=2)
    
    return {
        'filename': filename,
        'total': len(data),
        'updated_year': updated_count,
        'year_0_remaining': year_0_count,
        'wrong_split': len(wrong_split_items),
        'wrong_split_items': wrong_split_items
    }


def move_wrong_split_items(results_by_variant):
    """将错误split的items移动到正确的文件中"""
    moves = []
    
    # 按变体类型分组处理
    for variant_type, results in results_by_variant.items():
        # 收集所有需要移动的items
        items_to_move = defaultdict(list)  # target_split -> items
        
        for result in results:
            for item in result['wrong_split_items']:
                target_split = item.pop('_target_split', None)
                if target_split:
                    items_to_move[target_split].append(item)
        
        # 移动到正确的文件
        for target_split, items in items_to_move.items():
            target_file = VARIANTS_DIR / f"{target_split}_{variant_type}_variants.json"
            
            # 加载目标文件
            if target_file.exists():
                with open(target_file, 'r', encoding='utf-8') as f:
                    target_data = json.load(f)
            else:
                target_data = []
            
            # 去重：检查是否已存在
            existing_ids = {(it.get('doc_id'), it.get('original_text', '')[:50]) for it in target_data}
            new_items = []
            for item in items:
                key = (item.get('doc_id'), item.get('original_text', '')[:50])
                if key not in existing_ids:
                    new_items.append(item)
                    existing_ids.add(key)
            
            # 追加到目标文件
            if new_items:
                target_data.extend(new_items)
                with open(target_file, 'w', encoding='utf-8') as f:
                    json.dump(target_data, f, ensure_ascii=False, indent=2)
                
                moves.append({
                    'variant': variant_type,
                    'to_split': target_split,
                    'count': len(new_items)
                })
    
    return moves


def validate_json_files():
    """验证所有JSON文件格式"""
    variants_files = list(VARIANTS_DIR.glob("*_variants.json"))
    
    errors = []
    for vf in variants_files:
        try:
            with open(vf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                errors.append(f"{vf.name}: 不是列表格式")
                continue
            
            # 检查每个item的必需字段
            required_fields = ['original_text', 'variant_text', 'variant_type', 'doc_id', 'year']
            for i, item in enumerate(data):
                for field in required_fields:
                    if field not in item:
                        errors.append(f"{vf.name}[{i}]: 缺少字段 {field}")
                        break
                
                # 检查year是否为0
                if item.get('year', 0) == 0:
                    errors.append(f"{vf.name}[{i}]: year=0 (doc_id={item.get('doc_id', 'unknown')})")
        
        except json.JSONDecodeError as e:
            errors.append(f"{vf.name}: JSON格式错误 - {e}")
        except Exception as e:
            errors.append(f"{vf.name}: 读取错误 - {e}")
    
    return errors


def main():
    print("=" * 70)
    print("📊 S5变体文件Year修复和Split检查")
    print("=" * 70)
    
    # 加载数据
    print("\n1. 加载元数据...")
    doc_year_map = load_metadata()
    print(f"   加载了 {len(doc_year_map)} 个文档的year信息")
    
    print("\n2. 加载split索引...")
    split_index = load_split_index()
    print(f"   加载了 {len(split_index)} 个文档的split分配")
    
    # 获取所有变体文件
    variants_files = list(VARIANTS_DIR.glob("*_variants.json"))
    print(f"\n3. 找到 {len(variants_files)} 个变体文件")
    
    # 按变体类型分组
    variants_by_type = defaultdict(list)
    for vf in variants_files:
        parts = vf.name.replace('_variants.json', '').split('_')
        variant_type = '_'.join(parts[1:])  # 去掉split前缀
        variants_by_type[variant_type].append(vf)
    
    # 处理每个文件
    print("\n4. 处理变体文件...")
    results_by_variant = defaultdict(list)
    
    for variant_type, files in variants_by_type.items():
        print(f"\n   处理变体类型: {variant_type}")
        for vf in files:
            result = process_variants_file(vf, doc_year_map, split_index, variant_type)
            results_by_variant[variant_type].append(result)
            
            print(f"     {result['filename']}: "
                  f"total={result['total']}, "
                  f"updated_year={result['updated_year']}, "
                  f"wrong_split={result['wrong_split']}")
    
    # 移动错误split的items
    print("\n5. 移动错误split的items...")
    moves = move_wrong_split_items(results_by_variant)
    if moves:
        for move in moves:
            print(f"   移动 {move['count']} 条到 {move['to_split']}_{move['variant']}")
    else:
        print("   无需移动")
    
    # 验证JSON文件
    print("\n6. 验证JSON文件格式...")
    errors = validate_json_files()
    if errors:
        print(f"   ❌ 发现 {len(errors)} 个错误:")
        for error in errors[:20]:  # 只显示前20个
            print(f"      - {error}")
        if len(errors) > 20:
            print(f"      ... 还有 {len(errors) - 20} 个错误")
    else:
        print("   ✅ 所有JSON文件格式正确")
    
    # 最终统计
    print("\n" + "=" * 70)
    print("📊 最终统计")
    print("=" * 70)
    
    for vf in sorted(VARIANTS_DIR.glob("*_variants.json")):
        with open(vf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计year
        years = defaultdict(int)
        for item in data:
            year = item.get('year', 0)
            years[year] += 1
        
        year_0 = years.get(0, 0)
        non_zero_years = {k: v for k, v in years.items() if k != 0}
        
        print(f"\n{vf.name}: {len(data)}条")
        if year_0 > 0:
            print(f"  ⚠️  year=0: {year_0}条")
        if non_zero_years:
            year_str = ", ".join([f"{y}:{c}" for y, c in sorted(non_zero_years.items())])
            print(f"  ✅  year分布: {year_str}")
    
    print("\n" + "=" * 70)
    print("✅ 处理完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
