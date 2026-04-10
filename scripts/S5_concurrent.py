#!/usr/bin/env python3
"""
S5 Phase 3: AI 变体生成（3路并发架构）

架构特点:
- 3槽位并发: train/dev/test 并行处理
- 主力模型: siliconflow, deepseek, qwen3.5-flash
- 休假模型: zhipu (每15分钟轮换)
- Fallback: qwen3-max, kimi-k2.5 (错误触发, 9分钟限时)
"""

import os
import json
import time
import random
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import httpx
from dotenv import load_dotenv

load_dotenv("/home/aigc/aigc_checker/.env")

# ============ 配置 ============

@dataclass
class ModelConfig:
    name: str
    model: str
    timeout: int

# 主力模型 (只使用deepseek)
PRIMARY_MODELS = [
    ModelConfig("deepseek", "deepseek-chat", 60),
    ModelConfig("deepseek", "deepseek-chat", 60),
    ModelConfig("deepseek", "deepseek-chat", 60),
]

# 休假模型 (轮换)
REST_MODEL = ModelConfig("deepseek", "deepseek-chat", 60)

# Fallback模型 (错误触发)
FALLBACK_MODELS = [
    ModelConfig("deepseek", "deepseek-chat", 60),
]

# Provider配置
PROVIDER_CONFIG = {
    "siliconflow": {"key": "SILICONFLOW_API_KEY", "url": "SILICONFLOW_BASE_URL"},
    "deepseek": {"key": "DEEPSEEK_API_KEY", "url": "DEEPSEEK_BASE_URL"},
    "aliyun_qwen35_flash": {"key": "ALIYUN_API_KEY", "url": "ALIYUN_QWEN35_FLASH_BASE_URL"},
    "zhipu": {"key": "ZHIPU_API_KEY", "url": "ZHIPU_BASE_URL"},
    "aliyun_qwen3": {"key": "ALIYUN_API_KEY", "url": "ALIYUN_QWEN3_MAX_BASE_URL"},
    "moonshot": {"key": "MOONSHOT_API_KEY", "url": "MOONSHOT_BASE_URL"},
}

ROTATION_INTERVAL = 900  # 15分钟
FALLBACK_DURATION = 540  # 9分钟
ERROR_THRESHOLD = 3

# ServerChan汇报配置
REPORT_INTERVAL = 900  # 15分钟
FIRST_REPORT_DELAY = 300  # 5分钟后第一次汇报
SENDKEY = os.getenv("SERVERCHAN_SENDKEY", "")

EXCLUDED_ROLES = ["reference", "limitation", "acknowledgement", "appendix"]
REQUIRED_PARA_TYPE = "normal"

# ============ 变体类型 ============

VARIANT_TYPES = {
    "hard_human": {
        "temperature": 0.4,
        "prompt": """你是一位写作规范的中国MBA学生。请对以下段落进行轻微润色：

原文：
{text}

要求：
1. 只做 minimal 的修改
2. 保持原有的句式结构和用词习惯
3. 保留一些口语化表达
4. 不要过于完美和规整

润色后："""
    },
    "multi_round_rewrite": {
        "temperature": 0.7,
        "prompt": """请对以下段落进行两轮改写：

原文：
{text}

要求：
1. 第一轮：学术化、规范化
2. 第二轮：增加个人写作风格
3. 最终版本应介于AI生成和人类写作之间

最终改写后："""
    },
    "partial_rewrite": {
        "temperature": 0.6,
        "prompt": """请仅改写以下段落的部分句子：

原文：
{text}

要求：
1. 只改写2-3个关键句子
2. 其他句子保持原样
3. 保持段落整体连贯性

改写后："""
    },
    "back_translation_mix": {
        "temperature": 0.6,
        "prompt": """请先将以下中文段落翻译成英文，然后再翻译回中文：

原文：
{text}

要求：
1. 先翻译成英文
2. 再翻译回中文
3. 润色使表达自然流畅

回译润色后："""
    },
    "polish_manual_mix": {
        "temperature": 0.5,
        "prompt": """你是一位正在润色论文的学生。请对以下段落进行适度润色：

原文：
{text}

要求：
1. 修正明显的语法错误
2. 保持原有的表达习惯
3. 不要过度规整化
4. 保留一些个人写作特征

润色后："""
    },
    "standard_rewrite": {
        "temperature": 0.5,
        "prompt": """你是一位MBA学生，正在修改论文。请对以下段落进行标准改写：

原文：
{text}

要求：
1. 修正语法错误和不通顺的表达
2. 保持原有的学术风格和术语
3. 适当调整句式结构
4. 保持段落的核心意思不变

改写后："""
    },
}

# ============ Fallback管理器 ============

@dataclass
class SupplementTarget:
    """补充任务目标"""
    split: str
    variant: str
    target: int

class ServerChanReporter:
    """ServerChan汇报器"""
    
    def __init__(self, supplement_targets: List[SupplementTarget]):
        self.supplement_targets = supplement_targets
        self.start_time = time.time()
        self.output_dir = Path("/home/aigc/aigc_checker/phase3/05_ai_variants")
    
    def get_current_progress(self) -> Dict:
        """获取当前进度"""
        progress = []
        total_current = 0
        total_target = 0
        
        for target in self.supplement_targets:
            file_path = self.output_dir / f"{target.split}_{target.variant}_variants.json"
            current = 0
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        current = len(data)
                except:
                    pass
            
            # 计算该任务实际需要补充的数量
            file_path_base = self.output_dir / f"{target.split}_{target.variant}_variants.json"
            existing = 0
            if file_path_base.exists():
                try:
                    with open(file_path_base, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        existing = len(data)
                except:
                    pass
            
            # 如果当前已达到或超过目标，显示为完成
            if current >= target.target:
                pct = 100
            else:
                pct = (current / target.target * 100) if target.target > 0 else 0
            
            progress.append({
                'split': target.split,
                'variant': target.variant,
                'current': current,
                'target': target.target,
                'percentage': pct
            })
            total_current += current
            total_target += target.target
        
        elapsed = time.time() - self.start_time
        total_pct = (total_current / total_target * 100) if total_target > 0 else 0
        
        # 预估剩余时间
        if total_current > 0 and total_pct < 100:
            estimated_total = elapsed / (total_pct / 100)
            remaining = estimated_total - elapsed
        else:
            remaining = 0
        
        return {
            'elapsed_minutes': int(elapsed / 60),
            'progress': progress,
            'total_current': total_current,
            'total_target': total_target,
            'total_percentage': total_pct,
            'remaining_minutes': int(remaining / 60)
        }
    
    def send_report(self):
        """发送汇报"""
        if not SENDKEY:
            print("[Report] 未配置ServerChan SENDKEY，跳过汇报")
            return
        
        progress_data = self.get_current_progress()
        
        # 构建标题
        title = f"[S5补充] 进度{progress_data['total_percentage']:.0f}% | {progress_data['elapsed_minutes']}分钟"
        
        # 构建内容
        lines = [
            f"运行时间: {progress_data['elapsed_minutes']}分钟",
            f"总体进度: {progress_data['total_current']}/{progress_data['total_target']} ({progress_data['total_percentage']:.1f}%)",
            f"预估剩余: {progress_data['remaining_minutes']}分钟",
            "",
            "各目标进度:"
        ]
        
        for p in progress_data['progress']:
            status = "✓" if p['percentage'] >= 100 else f"{p['percentage']:.0f}%"
            lines.append(f"- {p['split']}/{p['variant']}: {p['current']}/{p['target']} ({status})")
        
        content = "\n".join(lines)
        
        try:
            url = f"https://sctapi.ftqq.com/{SENDKEY}.send"
            response = httpx.post(url, data={"title": title, "desp": content}, timeout=30)
            if response.status_code == 200:
                print(f"[Report] ServerChan汇报成功: {title}")
            else:
                print(f"[Report] ServerChan汇报失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"[Report] ServerChan汇报异常: {e}")

class FallbackManager:
    """Fallback状态管理器"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.fallback_until: Dict[str, float] = {}  # slot -> timestamp
        self.last_provider: Dict[str, str] = {}  # slot -> last used provider
    
    def record_error(self, provider: str, status_code: int) -> bool:
        """记录错误，返回是否触发fallback"""
        if 400 <= status_code < 600:
            self.error_counts[provider] += 1
            if self.error_counts[provider] >= ERROR_THRESHOLD:
                print(f"    [Fallback] {provider} 连续{ERROR_THRESHOLD}次错误，触发fallback")
                return True
        return False
    
    def activate_fallback(self, slot: str):
        """激活fallback模式"""
        self.fallback_until[slot] = time.time() + FALLBACK_DURATION
        print(f"    [Fallback] Slot {slot} 进入fallback模式，持续9分钟")
    
    def is_fallback_active(self, slot: str) -> bool:
        """检查slot是否处于fallback模式"""
        if slot not in self.fallback_until:
            return False
        if time.time() > self.fallback_until[slot]:
            print(f"    [Fallback] Slot {slot} fallback时间结束，返回主力模型")
            del self.fallback_until[slot]
            # 重置该slot之前使用的主力模型错误计数
            if slot in self.last_provider:
                self.error_counts[self.last_provider[slot]] = 0
            return False
        return True
    
    def reset_error_count(self, provider: str):
        """重置错误计数"""
        if self.error_counts[provider] > 0:
            print(f"    [Fallback] {provider} 重置错误计数")
            self.error_counts[provider] = 0
    
    def set_last_provider(self, slot: str, provider: str):
        """记录slot最后使用的主力provider"""
        self.last_provider[slot] = provider

# ============ 槽位管理器 ============

class SlotManager:
    """3槽位并发管理器"""
    
    def __init__(self):
        self.slots = ['train', 'dev', 'test']
        self.rotation_index = 0
        self.last_rotation = time.time()
        
        # 每个slot分配一个主力模型 (初始)
        self.slot_assignment = {
            'train': PRIMARY_MODELS[0],  # siliconflow
            'dev': PRIMARY_MODELS[1],     # deepseek
            'test': PRIMARY_MODELS[2],    # qwen3.5-flash
        }
    
    def should_rotate(self) -> bool:
        """检查是否需要轮换"""
        return time.time() - self.last_rotation >= ROTATION_INTERVAL
    
    def rotate(self):
        """执行15分钟轮换"""
        # 轮换策略: 将当前slot的主力模型替换为休假模型
        # 被替换的主力模型去休假
        slots = list(self.slot_assignment.keys())
        slot_to_rotate = slots[self.rotation_index % len(slots)]
        
        old_model = self.slot_assignment[slot_to_rotate]
        self.slot_assignment[slot_to_rotate] = REST_MODEL
        
        self.rotation_index += 1
        self.last_rotation = time.time()
        
        print(f"\n[Rotation] 15分钟轮换完成")
        print(f"  Slot '{slot_to_rotate}': {old_model.name} -> {REST_MODEL.name} (休假)")
        print(f"  当前分配:")
        for slot, model in self.slot_assignment.items():
            print(f"    {slot}: {model.name}")
    
    def get_provider_for_slot(self, slot: str, fallback_manager: FallbackManager) -> ModelConfig:
        """获取slot当前应使用的provider"""
        if fallback_manager.is_fallback_active(slot):
            # 从fallback池中选择一个
            idx = hash(slot + str(int(time.time()) // 60)) % len(FALLBACK_MODELS)
            return FALLBACK_MODELS[idx]
        
        return self.slot_assignment.get(slot, PRIMARY_MODELS[0])

# ============ API客户端 ============

class APIClient:
    """异步API客户端"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=120)
    
    async def call(self, provider: ModelConfig, prompt: str, temperature: float) -> Tuple[bool, str, int]:
        """调用API，返回(成功, 内容/错误, 状态码)"""
        config = PROVIDER_CONFIG.get(provider.name)
        if not config:
            return False, f"Unknown provider: {provider.name}", 0
        
        api_key = os.getenv(config["key"])
        base_url = os.getenv(config["url"])
        
        if not api_key or not base_url:
            return False, f"Missing config for {provider.name}", 0
        
        try:
            response = await self.client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": provider.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": temperature,
                },
                timeout=provider.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "").strip()
                    return True, content, 200
                return False, "Empty response", 200
            else:
                error = response.text[:200]
                return False, error, response.status_code
                
        except Exception as e:
            return False, str(e)[:200], 0
    
    async def close(self):
        await self.client.aclose()

# ============ 变体生成器 ============

class VariantGenerator:
    """变体生成器"""
    
    def __init__(self, output_dir: Path, supplement_targets: Optional[List[SupplementTarget]] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_client = APIClient()
        self.slot_manager = SlotManager()
        self.fallback_manager = FallbackManager()
        
        # 统计
        self.stats = defaultdict(lambda: defaultdict(int))
        
        # 补充模式
        self.supplement_targets = supplement_targets
        self.reporter = ServerChanReporter(supplement_targets) if supplement_targets else None
    
    def filter_para(self, para: Dict) -> bool:
        """过滤段落"""
        if para.get("para_type") != REQUIRED_PARA_TYPE:
            return False
        if para.get("chapter_role") in EXCLUDED_ROLES:
            return False
        return True
    
    def load_existing(self, variant_type: str, split: str) -> List[Dict]:
        """加载现有变体"""
        file_path = self.output_dir / f"{split}_{variant_type}_variants.json"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def save_variants(self, variant_type: str, split: str, variants: List[Dict]):
        """保存变体"""
        file_path = self.output_dir / f"{split}_{variant_type}_variants.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(variants, f, ensure_ascii=False, indent=2)
    
    async def generate_single(self, text: str, variant_type: str, slot: str) -> Optional[str]:
        """生成单个变体"""
        config = VARIANT_TYPES.get(variant_type)
        if not config:
            return None
        
        prompt = config["prompt"].format(text=text)
        temperature = config["temperature"]
        
        # 获取当前provider
        provider = self.slot_manager.get_provider_for_slot(slot, self.fallback_manager)
        
        print(f"      [{slot}] 尝试 {provider.name}...")
        
        success, result, status_code = await self.api_client.call(provider, prompt, temperature)
        
        if success:
            print(f"      [{slot}] ✓ {provider.name} 成功")
            self.fallback_manager.reset_error_count(provider.name)
            self.fallback_manager.set_last_provider(slot, provider.name)
            self.stats[slot]["success"] += 1
            self.stats[slot][provider.name] += 1
            return result
        else:
            print(f"      [{slot}] ✗ {provider.name} 失败: HTTP {status_code}")
            
            # 记录错误，检查是否触发fallback
            if self.fallback_manager.record_error(provider.name, status_code):
                self.fallback_manager.activate_fallback(slot)
                self.fallback_manager.set_last_provider(slot, provider.name)
                # 立即重试一次fallback
                fallback = self.slot_manager.get_provider_for_slot(slot, self.fallback_manager)
                print(f"      [{slot}] 立即尝试fallback: {fallback.name}...")
                success, result, _ = await self.api_client.call(fallback, prompt, temperature)
                if success:
                    print(f"      [{slot}] ✓ fallback {fallback.name} 成功")
                    self.stats[slot]["fallback_success"] += 1
                    return result
            
            self.stats[slot]["failed"] += 1
            return None
    
    async def process_split_variant(self, split: str, paragraphs: List[Dict], 
                                     variant_type: str, target: int):
        """处理单个split的单个变体类型"""
        print(f"\n  [{split}] {variant_type} 目标: {target}")
        
        # 过滤段落
        filtered = [p for p in paragraphs if self.filter_para(p)]
        
        # 加载现有
        existing = self.load_existing(variant_type, split)
        print(f"    已存在: {len(existing)}, 可用段落: {len(filtered)}")
        
        if len(existing) >= target:
            print(f"    ✓ 已达到目标，跳过")
            return
        
        # 准备处理
        existing_ids = {v.get("doc_id", "") for v in existing}
        to_process = []
        for para in filtered:
            doc_id = f"{para.get('doc_id', '')}_{para.get('para_index', 0)}"
            if doc_id not in existing_ids:
                to_process.append((doc_id, para))
        
        need = target - len(existing)
        to_process = to_process[:need]
        print(f"    需要处理: {len(to_process)}")
        
        results = existing.copy()
        
        for i, (doc_id, para) in enumerate(to_process):
            text = para.get("text", "")
            if len(text) < 50 or len(text) > 2000:
                continue
            
            print(f"\n    [{i+1}/{len(to_process)}] {doc_id[:50]}...")
            
            variant_text = await self.generate_single(text, variant_type, split)
            
            if variant_text:
                results.append({
                    "original_text": text,
                    "variant_text": variant_text,
                    "variant_type": variant_type,
                    "doc_id": para.get("doc_id"),
                    "year": para.get("year"),
                    "para_type": para.get("para_type"),
                    "chapter_role": para.get("chapter_role"),
                })
            
            # 每10个保存
            if (i + 1) % 10 == 0:
                self.save_variants(variant_type, split, results)
                print(f"    [Checkpoint] 已保存 {len(results)} 个")
            
            # 限速
            await asyncio.sleep(0.5)
        
        # 最终保存
        self.save_variants(variant_type, split, results)
        print(f"    [{split}] {variant_type} 完成: {len(results)}/{target}")
    
    async def rotation_monitor(self):
        """轮换监控循环"""
        while True:
            await asyncio.sleep(60)  # 每分钟检查
            if self.slot_manager.should_rotate():
                self.slot_manager.rotate()
    
    async def report_monitor(self):
        """汇报监控循环"""
        if not self.reporter:
            return
        
        # 先等5分钟发送第一次汇报
        print(f"[Report] 5分钟后发送第一次汇报...")
        await asyncio.sleep(FIRST_REPORT_DELAY)
        self.reporter.send_report()
        
        # 之后每15分钟汇报一次
        while True:
            await asyncio.sleep(REPORT_INTERVAL)
            self.reporter.send_report()
    
    async def run(self, variant_types: List[str], targets: Dict[str, int]):
        """运行生成"""
        print("="*60)
        if self.supplement_targets:
            print("S5 Concurrent - 补充模式")
            print("="*60)
            print(f"\n补充任务:")
            for t in self.supplement_targets:
                print(f"  {t.split}/{t.variant}: 目标{t.target}条")
        else:
            print("S5 Concurrent - 3路并发AI变体生成")
            print("="*60)
        
        # 加载数据
        print("\n加载数据...")
        splits = {}
        for split in ['train', 'dev', 'test']:
            file_path = Path(f"/home/aigc/aigc_checker/phase3/04_dataset_split/{split}.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                splits[split] = json.load(f)
            print(f"  {split}: {len(splits[split])} 段落")
        
        print(f"\n主力模型: {[m.name for m in PRIMARY_MODELS]}")
        print(f"休假模型: {REST_MODEL.name}")
        print(f"Fallback: {[m.name for m in FALLBACK_MODELS]}")
        print(f"轮换间隔: {ROTATION_INTERVAL/60}分钟")
        
        # 启动轮换监控
        rotation_task = asyncio.create_task(self.rotation_monitor())
        
        # 启动汇报监控（如果是补充模式）
        report_task = None
        if self.reporter:
            report_task = asyncio.create_task(self.report_monitor())
        
        # 为每个split创建处理任务
        tasks = []
        
        if self.supplement_targets:
            # 补充模式：只处理指定的任务
            for target in self.supplement_targets:
                split = target.split
                vtype = target.variant
                target_count = target.target
                task = self.process_split_variant(split, splits[split], vtype, target_count)
                tasks.append(task)
        else:
            # 正常模式：处理所有split和变体类型
            for split in ['train', 'dev', 'test']:
                for vtype in variant_types:
                    target = targets.get(vtype, 400)
                    task = self.process_split_variant(split, splits[split], vtype, target)
                    tasks.append(task)
        
        print(f"\n总共 {len(tasks)} 个任务")
        print("开始并发处理...\n")
        
        # 并发执行所有任务
        await asyncio.gather(*tasks)
        
        # 取消轮换监控
        rotation_task.cancel()
        try:
            await rotation_task
        except asyncio.CancelledError:
            pass
        
        # 取消汇报监控
        if report_task:
            report_task.cancel()
            try:
                await report_task
            except asyncio.CancelledError:
                pass
            # 发送最终汇报
            if self.reporter:
                self.reporter.send_report()
        
        # 打印统计
        print("\n" + "="*60)
        print("处理统计")
        print("="*60)
        for slot, stats in self.stats.items():
            print(f"\n  [{slot}]")
            for key, val in stats.items():
                print(f"    {key}: {val}")
        
        await self.api_client.close()
        print("\n✓ 所有任务完成")

# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant_types", nargs="+", 
                       default=["hard_human", "multi_round_rewrite", "partial_rewrite", 
                               "back_translation_mix", "polish_manual_mix"])
    parser.add_argument("--targets", type=str, 
                       default='{"hard_human":400,"multi_round_rewrite":350,"partial_rewrite":350,"back_translation_mix":300,"polish_manual_mix":400}')
    parser.add_argument("--supplement_mode", action="store_true",
                       help="启用补充模式，只处理supplement_targets指定的任务")
    parser.add_argument("--supplement_targets", type=str,
                       default='[]',
                       help='补充任务列表，格式: [{"split": "dev", "variant": "standard_rewrite", "target": 173}, ...]')
    args = parser.parse_args()
    
    targets = json.loads(args.targets)
    
    # 解析补充目标
    supplement_targets = None
    if args.supplement_mode and args.supplement_targets:
        targets_data = json.loads(args.supplement_targets)
        supplement_targets = [SupplementTarget(t['split'], t['variant'], t['target']) for t in targets_data]
    
    output_dir = Path("/home/aigc/aigc_checker/phase3/05_ai_variants")
    generator = VariantGenerator(output_dir, supplement_targets)
    
    asyncio.run(generator.run(args.variant_types, targets))

if __name__ == "__main__":
    main()
