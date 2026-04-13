#!/usr/bin/env python3
"""
AIGC检测报告生成器 - HTML全文标注版
生成带有颜色标注的完整文档检测报告
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse


class HTMLReportGenerator:
    """HTML报告生成器 - 全文标注版"""
    
    def __init__(self, result_data: Dict, docx_path: str, paragraphs: List[str]):
        self.data = result_data
        self.docx_path = Path(docx_path)
        self.paragraphs = paragraphs
        self.report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_id = f"MBA-AIGC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
    def generate(self, output_path: str):
        """生成HTML报告"""
        html = self._build_html()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ HTML报告已生成: {output_path}")
        
    def _build_html(self) -> str:
        """构建完整HTML文档"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MBA论文AIGC检测报告 - {self.data.get('risk_level', '未知')}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
        
        :root {{
            --primary: #1a365d;
            --primary-light: #2c5282;
            --accent: #c53030;
            --accent-light: #e53e3e;
            --warning: #d69e2e;
            --success: #276749;
            --bg-paper: #faf9f7;
            --bg-card: #ffffff;
            --text-primary: #1a202c;
            --text-secondary: #4a5568;
            --border: #e2e8f0;
            
            /* 风险等级颜色 */
            --risk-low-bg: #dcfce7;
            --risk-low-text: #166534;
            --risk-low-border: #86efac;
            
            --risk-medium-bg: #fef3c7;
            --risk-medium-text: #92400e;
            --risk-medium-border: #fcd34d;
            
            --risk-high-bg: #fee2e2;
            --risk-high-text: #991b1b;
            --risk-high-border: #fca5a5;
            
            --risk-critical-bg: #fecaca;
            --risk-critical-text: #7f1d1d;
            --risk-critical-border: #f87171;
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Noto Sans SC', -apple-system, sans-serif;
            background: var(--bg-paper);
            color: var(--text-primary);
            line-height: 1.8;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        
        /* 报告头部 */
        .report-header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 10px 40px rgba(26, 54, 93, 0.3);
        }}
        
        .report-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 1px;
            margin-bottom: 16px;
        }}
        
        .report-title {{
            font-family: 'Noto Serif SC', serif;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 20px;
        }}
        
        .report-meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            font-size: 13px;
            opacity: 0.9;
        }}
        
        /* 核心指标 */
        .metrics-bar {{
            display: flex;
            gap: 20px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}
        
        .metric-item {{
            flex: 1;
            min-width: 200px;
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid var(--primary);
        }}
        
        .metric-item.warning {{ border-left-color: var(--warning); }}
        .metric-item.danger {{ border-left-color: var(--accent); }}
        
        .metric-value {{
            font-family: 'Noto Serif SC', serif;
            font-size: 36px;
            font-weight: 700;
            color: var(--primary);
            line-height: 1;
        }}
        
        .metric-value.warning {{ color: var(--warning); }}
        .metric-value.danger {{ color: var(--accent); }}
        
        .metric-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .risk-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-top: 12px;
        }}
        
        .risk-badge.low {{ background: var(--risk-low-bg); color: var(--risk-low-text); }}
        .risk-badge.medium {{ background: var(--risk-medium-bg); color: var(--risk-medium-text); }}
        .risk-badge.high {{ background: var(--risk-high-bg); color: var(--risk-high-text); }}
        .risk-badge.critical {{ background: var(--risk-critical-bg); color: var(--risk-critical-text); }}
        
        /* 图例 */
        .legend-bar {{
            background: white;
            padding: 16px 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .legend-title {{
            font-weight: 600;
            color: var(--text-primary);
            margin-right: 12px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid transparent;
        }}
        
        /* 全文容器 */
        .document-container {{
            background: white;
            border-radius: 12px;
            padding: 48px 60px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            font-family: 'Noto Serif SC', 'SimSun', serif;
            font-size: 16px;
            line-height: 2;
        }}
        
        /* 段落样式 */
        .para {{
            margin-bottom: 16px;
            padding: 16px 20px;
            border-radius: 8px;
            border-left: 4px solid transparent;
            position: relative;
            transition: all 0.2s ease;
        }}
        
        .para:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transform: translateX(4px);
        }}
        
        /* 风险等级样式 */
        .para.low {{
            background: var(--risk-low-bg);
            border-left-color: var(--risk-low-border);
        }}
        
        .para.medium {{
            background: var(--risk-medium-bg);
            border-left-color: var(--risk-medium-border);
        }}
        
        .para.high {{
            background: var(--risk-high-bg);
            border-left-color: var(--risk-high-border);
        }}
        
        .para.critical {{
            background: var(--risk-critical-bg);
            border-left-color: var(--risk-critical-border);
        }}
        
        .para-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-family: 'Noto Sans SC', sans-serif;
        }}
        
        .para-num {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            background: rgba(255,255,255,0.6);
            padding: 4px 10px;
            border-radius: 12px;
        }}
        
        .para-score {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            font-weight: 600;
        }}
        
        .para-score.low {{ color: var(--risk-low-text); }}
        .para-score.medium {{ color: var(--risk-medium-text); }}
        .para-score.high {{ color: var(--risk-high-text); }}
        .para-score.critical {{ color: var(--risk-critical-text); }}
        
        .para-text {{
            text-indent: 2em;
            color: var(--text-primary);
        }}
        
        /* 统计面板 */
        .stats-panel {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .stat-item {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 4px;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: var(--text-secondary);
        }}
        
        /* 建议区域 */
        .suggestions {{
            background: white;
            border-radius: 12px;
            padding: 32px;
            margin-top: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }}
        
        .suggestions h2 {{
            font-family: 'Noto Serif SC', serif;
            font-size: 20px;
            color: var(--primary);
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--border);
        }}
        
        .suggestion-list {{
            list-style: none;
        }}
        
        .suggestion-list li {{
            padding: 16px 0;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 16px;
        }}
        
        .suggestion-num {{
            width: 28px;
            height: 28px;
            background: var(--primary);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 600;
            flex-shrink: 0;
        }}
        
        .suggestion-content h4 {{
            font-weight: 600;
            margin-bottom: 4px;
        }}
        
        .suggestion-content p {{
            font-size: 14px;
            color: var(--text-secondary);
        }}
        
        /* 免责声明 */
        .disclaimer {{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-top: 24px;
            text-align: center;
            font-size: 13px;
            color: var(--text-secondary);
        }}
        
        /* 打印样式 */
        @media print {{
            .container {{ padding: 0; }}
            .document-container {{ box-shadow: none; }}
            .para {{ break-inside: avoid; }}
        }}
        
        @media (max-width: 768px) {{
            .document-container {{ padding: 24px; }}
            .metrics-bar {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {self._render_header()}
        {self._render_metrics()}
        {self._render_stats()}
        {self._render_legend()}
        {self._render_full_text()}
        {self._render_suggestions()}
        {self._render_footer()}
    </div>
</body>
</html>"""
    
    def _render_header(self) -> str:
        """渲染报告头部"""
        filename = self.docx_path.name
        return f"""
        <header class="report-header">
            <div class="report-badge">AIGC检测全文报告 · CONFIDENTIAL</div>
            <h1 class="report-title">MBA论文AIGC检测报告</h1>
            <div class="report-meta">
                <div>报告编号: {self.report_id}</div>
                <div>检测时间: {self.report_time}</div>
                <div>文件名: {filename}</div>
                <div>检测系统: MBA-AIGC-Detector v1.0 (CNKI校准版)</div>
            </div>
        </header>"""
    
    def _render_metrics(self) -> str:
        """渲染核心指标"""
        score = self.data.get('doc_calibrated_score', 0) * 100
        raw_score = self.data.get('doc_raw_score', 0) * 100
        risk = self.data.get('risk_level', '未知')
        
        if '低' in risk:
            risk_class = 'low'
            card_class = ''
        elif '中' in risk:
            risk_class = 'medium'
            card_class = 'warning'
        elif '较高' in risk:
            risk_class = 'high'
            card_class = 'danger'
        else:
            risk_class = 'critical'
            card_class = 'danger'
        
        return f"""
        <div class="metrics-bar">
            <div class="metric-item {card_class}">
                <div class="metric-label">文档级AIGC分数 (CNKI校准)</div>
                <div class="metric-value {card_class}">{score:.1f}%</div>
                <span class="risk-badge {risk_class}">{risk}</span>
            </div>
            <div class="metric-item">
                <div class="metric-label">原始检测分数</div>
                <div class="metric-value">{raw_score:.1f}%</div>
            </div>
        </div>"""
    
    def _render_stats(self) -> str:
        """渲染统计面板"""
        ai_count = self.data.get('ai_paragraph_count', 0)
        total = self.data.get('total_paragraphs', 0)
        human_count = total - ai_count
        ratio = self.data.get('ai_ratio', 0) * 100
        
        # 计算各风险等级段落数
        paras = self.data.get('paragraph_results', [])
        low_count = sum(1 for p in paras if p['calibrated_score'] * 100 < 15)
        medium_count = sum(1 for p in paras if 15 <= p['calibrated_score'] * 100 < 30)
        high_count = sum(1 for p in paras if 30 <= p['calibrated_score'] * 100 < 50)
        critical_count = sum(1 for p in paras if p['calibrated_score'] * 100 >= 50)
        
        return f"""
        <div class="stats-panel">
            <div class="stat-item">
                <div class="stat-value">{total}</div>
                <div class="stat-label">总段落数</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color: var(--accent);">{ai_count}</div>
                <div class="stat-label">AI疑似段落</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color: var(--success);">{human_count}</div>
                <div class="stat-label">人工疑似段落</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color: var(--warning);">{medium_count + high_count + critical_count}</div>
                <div class="stat-label">中高风险段落</div>
            </div>
        </div>"""
    
    def _render_legend(self) -> str:
        """渲染图例"""
        return """
        <div class="legend-bar">
            <span class="legend-title">📊 段落风险等级标注:</span>
            <div class="legend-item">
                <div class="legend-color" style="background: var(--risk-low-bg); border-color: var(--risk-low-border);"></div>
                <span>低风险 (<15%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: var(--risk-medium-bg); border-color: var(--risk-medium-border);"></div>
                <span>中风险 (15-30%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: var(--risk-high-bg); border-color: var(--risk-high-border);"></div>
                <span>较高风险 (30-50%)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: var(--risk-critical-bg); border-color: var(--risk-critical-border);"></div>
                <span>高风险 (>50%)</span>
            </div>
        </div>"""
    
    def _render_full_text(self) -> str:
        """渲染带标注的全文"""
        results = self.data.get('paragraph_results', [])
        
        # 创建段落ID到检测结果的映射
        result_map = {p['para_id']: p for p in results}
        
        paras_html = []
        for i, text in enumerate(self.paragraphs):
            if not text.strip():
                continue
                
            result = result_map.get(i, {})
            score = result.get('calibrated_score', 0) * 100
            is_ai = result.get('is_ai', False)
            triggered = result.get('triggered_by', 'N/A')
            
            # 确定风险等级
            if score < 15:
                risk_class = 'low'
                risk_text = '低风险'
            elif score < 30:
                risk_class = 'medium'
                risk_text = '中风险'
            elif score < 50:
                risk_class = 'high'
                risk_text = '较高风险'
            else:
                risk_class = 'critical'
                risk_text = '高风险'
            
            # 截断过长的文本用于显示
            display_text = text[:500] + ('...' if len(text) > 500 else '')
            
            paras_html.append(f"""
            <div class="para {risk_class}">
                <div class="para-header">
                    <span class="para-num">段落 {i+1}</span>
                    <span class="para-score {risk_class}">
                        AIGC: {score:.1f}% | {risk_text} | 模型: {triggered}
                    </span>
                </div>
                <div class="para-text">{display_text}</div>
            </div>""")
        
        return f"""
        <div class="document-container">
            <h2 style="font-family: 'Noto Sans SC', sans-serif; font-size: 18px; color: var(--primary); margin-bottom: 24px; padding-bottom: 12px; border-bottom: 2px solid var(--border);">
                📄 检测文档全文 (带风险标注)
            </h2>
            {''.join(paras_html)}
        </div>"""
    
    def _render_suggestions(self) -> str:
        """渲染建议"""
        score = self.data.get('doc_calibrated_score', 0) * 100
        
        if score < 15:
            risk_desc = "低风险"
            suggestions = [
                ("整体评估良好", "文档AIGC分数在正常范围内，未发现明显的AI生成痕迹。"),
                ("继续保持", "建议保持当前的写作风格，注重个人观点和原创性表达。"),
            ]
        elif score < 30:
            risk_desc = "中风险"
            suggestions = [
                ("重点复核", "建议对黄色标注(中风险)的段落进行人工审查和改写。"),
                ("增加原创性", "在疑似AI段落中增加个人观点、案例分析、实地调研数据。"),
                ("改写优化", "对模板化表达进行改写，使用更自然的语言风格。"),
            ]
        elif score < 50:
            risk_desc = "较高风险"
            suggestions = [
                ("深度审查", "文档存在明显的AI生成痕迹，建议进行全面审查和改写。"),
                ("重构内容", "对橙色/红色标注段落进行深度改写，增加原创分析。"),
                ("数据支撑", "补充原始调研数据、访谈记录等一手资料。"),
            ]
        else:
            risk_desc = "高风险"
            suggestions = [
                ("紧急处理", "文档高度疑似AI生成，建议暂停提交，进行全面重写。"),
                ("重新撰写", "建议基于个人研究重新撰写论文，确保学术诚信。"),
                ("导师沟通", "立即与导师沟通，说明情况并寻求指导。"),
            ]
        
        items = ''.join([
            f"""
            <li>
                <div class="suggestion-num">{i+1}</div>
                <div class="suggestion-content">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </li>"""
            for i, (title, desc) in enumerate(suggestions)
        ])
        
        return f"""
        <div class="suggestions">
            <h2>💡 修改建议 ({risk_desc})</h2>
            <ul class="suggestion-list">
                {items}
            </ul>
        </div>"""
    
    def _render_footer(self) -> str:
        """渲染页脚"""
        return f"""
        <div class="disclaimer">
            <p><strong>免责声明</strong></p>
            <p>本系统输出仅供参考，不构成对学术不端行为的直接认定，也不替代学校官方检测系统或CNKI等权威检测结果。</p>
            <p style="margin-top: 8px;">MBA-AIGC-Detector v1.0 | 报告生成时间: {self.report_time}</p>
        </div>"""


def main():
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='生成HTML格式AIGC检测报告(全文标注版)')
    parser.add_argument('--json', type=str, required=True, help='检测结果JSON文件路径')
    parser.add_argument('--docx', type=str, required=True, help='原始DOCX文件路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出HTML文件路径')
    
    args = parser.parse_args()
    
    # 读取检测结果
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 读取文档段落
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor()
    paragraphs = processor.process(Path(args.docx))
    
    # 生成报告
    generator = HTMLReportGenerator(data, args.docx, paragraphs)
    generator.generate(args.output)


if __name__ == '__main__':
    main()
