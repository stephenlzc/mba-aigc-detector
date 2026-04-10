#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1.5 Phase 3: Markdown 质量控制（保守修正版）

核心调整（相对于主线 S1.5）：
- 降低"所有英文段都调用模型修正"的倾向
- 增加对章节标题、摘要、表格说明、图注的保守策略
- 只有真正脏段才调 AI 修正（结构性问题优先）
- 保留更多原始人类写作风格

输入: 01_markdown_output/
输出: phase3/01_qc_output/ (同时备份原始文件结构)
"""

import os
import re
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============ 配置 ============

QC_STANDARDS = {
    "garbled_ratio_threshold": 0.01,
    "chinese_ratio_threshold": 0.30,
    "min_char_count": 1000,
}

# 保守策略：这些类型的段落不轻易调用模型修正
CONSERVATIVE_PARA_TYPES = [
    "heading",        # 章节标题
    "abstract",       # 摘要
    "table_note",     # 表格说明
    "figure_note",    # 图注
    "formula_note",   # 公式说明
    "reference",      # 参考文献
]

# 真正需要修正的脏段模式（结构性问题）
DIRTY_PATTERNS = {
    "severe_letter_spacing": re.compile(r'[A-Za-z](?:\s+[A-Za-z]){4,}'),  # 5+字母被空格分隔
    "high_garbled": re.compile(r'[\ufffd\x00-\x08]{3,}'),  # 3+乱码字符
    "broken_url": re.compile(r'https?://\s+'),  # URL被断开
    "repeated_punct": re.compile(r'([。！？；])\1{2,}'),  # 重复标点
}


# ============ 数据结构 ============

@dataclass
class QualityIssue:
    issue_type: str
    description: str
    location: str
    severity: str

@dataclass
class QualityReport:
    file_path: str
    relative_path: str
    status: str
    char_count: int
    chinese_ratio: float
    garbled_ratio: float
    has_chapter_structure: bool
    has_valid_ending: bool
    issues: List[QualityIssue]
    fixes_applied: List[str]
    needs_model_fix: bool


# ============ 规则清洗函数（保守版） ============

def fix_letter_spacing_conservative(text: str) -> Tuple[str, bool]:
    """
    保守修复字母间距异常
    只修复严重的模式（4+字母被空格分隔）
    """
    original = text
    changed = False
    
    # 只修复严重的字母间距问题（4+字母）
    pattern = r'\b([A-Za-z](?:\s[A-Za-z]){3,})\b'
    
    def merge_match(m):
        word = m.group(1)
        if ' ' in word:
            return ''.join(word.split())
        return word
    
    new_text = re.sub(pattern, merge_match, text)
    if new_text != text:
        text = new_text
        changed = True
    
    # 清理中文字符与英文/数字间多余空格
    text = re.sub(r'([\u4e00-\u9fff])\s+([a-zA-Z0-9])', r'\1\2', text)
    text = re.sub(r'([a-zA-Z0-9])\s+([\u4e00-\u9fff])', r'\1\2', text)
    
    return text, changed


def remove_page_headers_footers(text: str) -> Tuple[str, bool]:
    """删除页眉页脚残留"""
    original = text
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 跳过页眉页脚模式
        if re.match(r'^第\s*\d+\s*页$', stripped):
            continue
        if re.match(r'^共\s*\d+\s*页$', stripped):
            continue
        # 纯数字行（页码）
        if re.match(r'^\d+$', stripped) and len(stripped) < 5:
            continue
        # 罗马数字页码
        if re.match(r'^[IVXLC]+$', stripped):
            continue
        # URL 行（CNKI 水印等）
        if re.match(r'^https?://', stripped):
            continue
        # 学校/机构水印
        if re.match(r'^.{0,10}大学.{0,10}硕士学位论文$', stripped):
            continue
            
        cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    return result, result != original


def fix_garbled_chars(text: str) -> Tuple[str, bool]:
    """清理乱码字符"""
    original = text
    
    # 移除 HTML 实体残留
    text = re.sub(r'&#x27;', "'", text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    
    # 修复 Unicode 替换字符
    text = text.replace('\ufffd', '')
    
    # 清理奇怪的标点组合
    text = re.sub(r'([。！？；])\s*([。！？；])', r'\1', text)
    
    return text, text != original


def clean_extra_spaces(text: str) -> Tuple[str, bool]:
    """清理多余空格"""
    original = text
    
    # 多个空格合并为一个（保留换行）
    text = re.sub(r' {2,}', ' ', text)
    
    # 清理换行符周围的多余空格
    text = re.sub(r' \n', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    
    # 最多保留两个连续换行（段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text, text != original


def stage1_rule_cleaning(text: str) -> Tuple[str, List[str]]:
    """
    Stage 1: 规则自动清洗（保守版）
    """
    fixes_applied = []
    
    operations = [
        ("页眉页脚删除", remove_page_headers_footers),
        ("字母间距修复", fix_letter_spacing_conservative),
        ("乱码清理", fix_garbled_chars),
        ("多余空格清理", clean_extra_spaces),
    ]
    
    for name, func in operations:
        text, changed = func(text)
        if changed:
            fixes_applied.append(name)
    
    return text, fixes_applied


# ============ 质量检测函数 ============

def calculate_chinese_ratio(text: str) -> float:
    """计算中文比例"""
    if not text:
        return 0.0
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.strip())
    return chinese_chars / total_chars if total_chars > 0 else 0.0


def calculate_garbled_ratio(text: str) -> float:
    """计算乱码比例"""
    if not text:
        return 0.0
    
    valid_pattern = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'
                               r'a-zA-Z0-9\s.,!?;:"\'()\[\]{}。，、！？；：「」『』（）'
                               r'【】《》〈〉—…·\-–—_/\\+=*@#$%^&|~`<>]')
    
    chars = list(text)
    if not chars:
        return 0.0
    
    valid_count = sum(1 for c in chars if valid_pattern.match(c))
    return 1 - (valid_count / len(chars))


def check_chapter_structure(text: str) -> bool:
    """检查章节结构"""
    patterns = [
        r'第[一二三四五六七八九十\d]+章',
        r'^#{1,6}\s+.*章',
        r'^绪论',
        r'^导论',
        r'^引言',
        r'^绪言',
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def check_valid_ending(text: str) -> bool:
    """检查正文结尾"""
    patterns = [r'参考文献', r'致谢', r'附录']
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def detect_para_type(text: str) -> str:
    """检测段落类型（用于保守策略）"""
    text_stripped = text.strip()
    
    # 章节标题
    if re.match(r'^#{1,6}\s+', text_stripped):
        return "heading"
    if re.match(r'^第[一二三四五六七八九十\d]+章', text_stripped):
        return "heading"
    
    # 摘要
    if re.match(r'^摘要\s*$', text_stripped, re.IGNORECASE):
        return "abstract"
    
    # 表格说明
    if re.match(r'^表[\s\d]', text_stripped):
        return "table_note"
    
    # 图注
    if re.match(r'^图[\s\d]', text_stripped):
        return "figure_note"
    
    # 公式说明
    if re.match(r'^公式[\s\d]', text_stripped) or re.match(r'^式\(', text_stripped):
        return "formula_note"
    
    # 参考文献
    if re.match(r'^参考文献', text_stripped):
        return "reference"
    
    return "normal"


def needs_model_fix(text: str) -> bool:
    """
    判断是否需要模型修正（保守策略）
    只有真正脏的段才需要
    """
    # 检查结构性脏段模式
    for name, pattern in DIRTY_PATTERNS.items():
        if pattern.search(text):
            return True
    
    # 乱码比例过高
    if calculate_garbled_ratio(text) > 0.05:
        return True
    
    return False


# ============ 质控主流程 ============

class MarkdownQualityControlPhase3:
    """Phase 3 Markdown 质量控制器（保守版）"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.all_reports: List[QualityReport] = []
        
    def get_markdown_files(self) -> List[Path]:
        """获取所有 Markdown 文件"""
        md_files = list(self.input_dir.rglob("*.md"))
        # 排除记录文件
        md_files = [f for f in md_files if "S1.5" not in f.name and "qc_" not in f.name]
        return sorted(md_files)
    
    def check_file(self, text: str, relative_path: str) -> QualityReport:
        """检查文件质量"""
        char_count = len(text)
        chinese_ratio = calculate_chinese_ratio(text)
        garbled_ratio = calculate_garbled_ratio(text)
        has_chapter = check_chapter_structure(text)
        has_ending = check_valid_ending(text)
        
        issues = []
        
        if garbled_ratio > QC_STANDARDS["garbled_ratio_threshold"]:
            issues.append(QualityIssue(
                "garbled_ratio",
                f"乱码比例 {garbled_ratio:.2%}",
                "全文",
                "error" if garbled_ratio > 0.05 else "warning"
            ))
        
        if chinese_ratio < QC_STANDARDS["chinese_ratio_threshold"]:
            issues.append(QualityIssue(
                "low_chinese_ratio",
                f"中文比例 {chinese_ratio:.2%}",
                "全文",
                "warning"
            ))
        
        if char_count < QC_STANDARDS["min_char_count"]:
            issues.append(QualityIssue(
                "low_char_count",
                f"字数 {char_count}",
                "全文",
                "warning"
            ))
        
        if not has_chapter:
            issues.append(QualityIssue(
                "missing_chapter_structure",
                "缺少章节结构",
                "全文",
                "warning"
            ))
        
        if not has_ending:
            issues.append(QualityIssue(
                "invalid_ending",
                "缺少参考文献/致谢/附录",
                "全文",
                "warning"
            ))
        
        # 判断是否需要模型修正（保守策略）
        needs_fix = needs_model_fix(text)
        
        if issues:
            has_errors = any(i.severity == "error" for i in issues)
            status = "error" if has_errors else "warning"
        else:
            status = "passed"
        
        return QualityReport(
            file_path="",
            relative_path=relative_path,
            status=status,
            char_count=char_count,
            chinese_ratio=chinese_ratio,
            garbled_ratio=garbled_ratio,
            has_chapter_structure=has_chapter,
            has_valid_ending=has_ending,
            issues=issues,
            fixes_applied=[],
            needs_model_fix=needs_fix
        )
    
    def process_file(self, md_path: Path) -> QualityReport:
        """处理单个文件"""
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            return QualityReport(
                file_path=str(md_path),
                relative_path=md_path.relative_to(self.input_dir).as_posix(),
                status="error",
                char_count=0,
                chinese_ratio=0.0,
                garbled_ratio=0.0,
                has_chapter_structure=False,
                has_valid_ending=False,
                issues=[QualityIssue("file_read_error", str(e), "N/A", "error")],
                fixes_applied=[],
                needs_model_fix=False
            )
        
        # Stage 1: 规则清洗
        cleaned_text, fixes_applied = stage1_rule_cleaning(text)
        
        # 计算相对路径
        relative_path = md_path.relative_to(self.input_dir)
        
        # 质量检测
        report = self.check_file(cleaned_text, relative_path.as_posix())
        report.file_path = str(md_path)
        report.fixes_applied = fixes_applied
        
        # 保存清洗后的文件
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        if fixes_applied:
            report.status = "fixed"
        
        return report
    
    def run(self, max_workers: int = 4) -> List[QualityReport]:
        """批量处理"""
        md_files = self.get_markdown_files()
        
        if not md_files:
            logging.warning(f"在 {self.input_dir} 中没有找到 Markdown 文件")
            return []
        
        logging.info(f"找到 {len(md_files)} 个 Markdown 文件")
        
        reports = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, f): f for f in md_files}
            
            for future in as_completed(future_to_file):
                md_path = future_to_file[future]
                try:
                    report = future.result()
                    reports.append(report)
                    
                    status_icon = {"passed": "✓", "warning": "⚠", "fixed": "✑", "error": "✗"}
                    icon = status_icon.get(report.status, "?")
                    logging.info(f"{icon} {report.relative_path}: {report.status} "
                                f"(字={report.char_count}, 中={report.chinese_ratio:.0%}, "
                                f"需模型修正={report.needs_model_fix})")
                except Exception as e:
                    logging.error(f"处理失败 {md_path}: {e}")
        
        self.all_reports = reports
        self._save_reports()
        
        # 统计
        passed = sum(1 for r in reports if r.status == "passed")
        warning = sum(1 for r in reports if r.status == "warning")
        fixed = sum(1 for r in reports if r.status == "fixed")
        error = sum(1 for r in reports if r.status == "error")
        needs_model = sum(1 for r in reports if r.needs_model_fix)
        
        logging.info("=" * 60)
        logging.info("S1.5 Phase 3 质控完成统计:")
        logging.info(f"  总计: {len(reports)} 个文件")
        logging.info(f"  通过: {passed}")
        logging.info(f"  警告: {warning}")
        logging.info(f"  已修正: {fixed}")
        logging.info(f"  错误: {error}")
        logging.info(f"  需模型修正: {needs_model}")
        logging.info("=" * 60)
        
        return reports
    
    def _save_reports(self):
        """保存报告"""
        if not self.all_reports:
            return
        
        report_data = []
        for r in self.all_reports:
            report_data.append({
                "file": r.file_path,
                "relative_path": r.relative_path,
                "status": r.status,
                "char_count": r.char_count,
                "chinese_ratio": round(r.chinese_ratio, 4),
                "garbled_ratio": round(r.garbled_ratio, 4),
                "has_chapter_structure": r.has_chapter_structure,
                "has_valid_ending": r.has_valid_ending,
                "issues": [{"type": i.issue_type, "desc": i.description,
                           "location": i.location, "severity": i.severity}
                          for i in r.issues],
                "fixes_applied": r.fixes_applied,
                "needs_model_fix": r.needs_model_fix
            })
        
        report_path = self.output_dir / "S1.5_qc_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "input_dir": str(self.input_dir),
                "total": len(report_data),
                "reports": report_data
            }, f, ensure_ascii=False, indent=2)
        
        logging.info(f"详细报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="S1.5 Phase 3: Markdown 质量控制（保守版）")
    parser.add_argument("--input_dir", type=str, default="/home/aigc/aigc_checker/01_markdown_output",
                        help="Markdown 输入目录")
    parser.add_argument("--output_dir", type=str, default="/home/aigc/aigc_checker/phase3/01_qc_output",
                        help="输出目录")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="并行处理的最大线程数")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    qc = MarkdownQualityControlPhase3(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    qc.run(max_workers=args.max_workers)


if __name__ == "__main__":
    main()
