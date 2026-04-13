#!/usr/bin/env python3
"""
AIGC检测报告生成器 - HTML可视化版
生成类似知网报告的专业检测报告
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import argparse


class HTMLReportGenerator:
    """HTML报告生成器"""
    
    def __init__(self, result_data: Dict, docx_path: str):
        self.data = result_data
        self.docx_path = Path(docx_path)
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-paper);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        
        /* 报告头部 */
        .report-header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: white;
            padding: 48px;
            border-radius: 16px;
            margin-bottom: 32px;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }}
        
        .report-header::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            border-radius: 50%;
        }}
        
        .header-content {{
            position: relative;
            z-index: 1;
        }}
        
        .report-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 1px;
            margin-bottom: 16px;
            backdrop-filter: blur(10px);
        }}
        
        .report-title {{
            font-family: 'Noto Serif SC', serif;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 24px;
        }}
        
        .report-meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .meta-label {{
            opacity: 0.7;
        }}
        
        /* 核心指标卡片区 */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 32px;
        }}
        
        .metric-card {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 32px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }}
        
        .metric-card.primary {{
            background: linear-gradient(135deg, #fef3f2 0%, #fff5f5 100%);
            border-color: #fecaca;
        }}
        
        .metric-card.success {{
            background: linear-gradient(135deg, #f0fdf4 0%, #f0fdf4 100%);
            border-color: #bbf7d0;
        }}
        
        .metric-card.warning {{
            background: linear-gradient(135deg, #fffbeb 0%, #fffbeb 100%);
            border-color: #fde68a;
        }}
        
        .metric-label {{
            font-size: 14px;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}
        
        .metric-value {{
            font-family: 'Noto Serif SC', serif;
            font-size: 48px;
            font-weight: 700;
            color: var(--accent);
            line-height: 1;
        }}
        
        .metric-value.success {{
            color: var(--success);
        }}
        
        .metric-value.warning {{
            color: var(--warning);
        }}
        
        .metric-desc {{
            margin-top: 12px;
            font-size: 14px;
            color: var(--text-secondary);
        }}
        
        /* 风险等级指示器 */
        .risk-indicator {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 16px;
        }}
        
        .risk-indicator.low {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .risk-indicator.medium {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .risk-indicator.high {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        .risk-indicator.critical {{
            background: #fecaca;
            color: #7f1d1d;
        }}
        
        /* 内容区块 */
        .section {{
            background: var(--bg-card);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
        }}
        
        .section-title {{
            font-family: 'Noto Serif SC', serif;
            font-size: 20px;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 24px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--border);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .section-icon {{
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 16px;
        }}
        
        /* 分布图表容器 */
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 24px 0;
        }}
        
        /* 段落检测结果表格 */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }}
        
        .data-table th {{
            background: var(--bg-paper);
            padding: 14px 16px;
            text-align: left;
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--border);
        }}
        
        .data-table td {{
            padding: 16px;
            border-bottom: 1px solid var(--border);
            font-size: 14px;
        }}
        
        .data-table tr:hover {{
            background: var(--bg-paper);
        }}
        
        .score-badge {{
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 600;
        }}
        
        .score-badge.low {{
            background: #dcfce7;
            color: #166534;
        }}
        
        .score-badge.medium {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .score-badge.high {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        /* 模型标签 */
        .model-tag {{
            display: inline-block;
            padding: 4px 10px;
            background: #e0e7ff;
            color: #3730a3;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        /* 说明文字 */
        .info-box {{
            background: linear-gradient(135deg, #eff6ff 0%, #eff6ff 100%);
            border-left: 4px solid var(--primary);
            padding: 20px;
            border-radius: 0 12px 12px 0;
            margin: 24px 0;
        }}
        
        .info-box h4 {{
            color: var(--primary);
            margin-bottom: 8px;
            font-size: 14px;
        }}
        
        .info-box p {{
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.8;
        }}
        
        /* 建议列表 */
        .suggestion-list {{
            list-style: none;
            padding: 0;
        }}
        
        .suggestion-list li {{
            padding: 16px 0;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: flex-start;
            gap: 16px;
        }}
        
        .suggestion-list li:last-child {{
            border-bottom: none;
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
        
        .suggestion-content {{
            flex: 1;
        }}
        
        .suggestion-title {{
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }}
        
        .suggestion-desc {{
            font-size: 14px;
            color: var(--text-secondary);
        }}
        
        /* 页脚 */
        .report-footer {{
            text-align: center;
            padding: 32px;
            color: var(--text-secondary);
            font-size: 13px;
            border-top: 1px solid var(--border);
            margin-top: 48px;
        }}
        
        /* 打印样式 */
        @media print {{
            .container {{
                padding: 20px;
            }}
            .section {{
                break-inside: avoid;
            }}
        }}
        
        /* 响应式 */
        @media (max-width: 768px) {{
            .report-header {{
                padding: 24px;
            }}
            .report-title {{
                font-size: 24px;
            }}
            .metric-value {{
                font-size: 36px;
            }}
            .data-table {{
                font-size: 12px;
            }}
            .data-table th,
            .data-table td {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {self._render_header()}
        {self._render_metrics()}
        {self._render_distribution()}
        {self._render_paragraph_details()}
        {self._render_suggestions()}
        {self._render_footer()}
    </div>
    
    <script>
        {self._render_charts_js()}
    </script>
</body>
</html>"""
    
    def _render_header(self) -> str:
        """渲染报告头部"""
        filename = self.docx_path.name
        return f"""
        <header class="report-header">
            <div class="header-content">
                <div class="report-badge">CONFIDENTIAL REPORT</div>
                <h1 class="report-title">MBA论文AIGC检测全文报告</h1>
                <div class="report-meta">
                    <div class="meta-item">
                        <span class="meta-label">报告编号:</span>
                        <span>{self.report_id}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">检测时间:</span>
                        <span>{self.report_time}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">文件名:</span>
                        <span>{filename}</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">检测系统:</span>
                        <span>MBA-AIGC-Detector v1.0 (CNKI校准版)</span>
                    </div>
                </div>
            </div>
        </header>"""
    
    def _render_metrics(self) -> str:
        """渲染核心指标卡片"""
        score = self.data.get('doc_calibrated_score', 0) * 100
        raw_score = self.data.get('doc_raw_score', 0) * 100
        ai_count = self.data.get('ai_paragraph_count', 0)
        total = self.data.get('total_paragraphs', 0)
        ratio = self.data.get('ai_ratio', 0) * 100
        risk = self.data.get('risk_level', '未知')
        
        # 确定风险等级样式
        if '低' in risk:
            risk_class = 'low'
        elif '中' in risk:
            risk_class = 'medium'
        elif '较高' in risk:
            risk_class = 'high'
        else:
            risk_class = 'critical'
        
        return f"""
        <div class="metrics-grid">
            <div class="metric-card primary">
                <div class="metric-label">文档级AIGC分数 (CNKI校准)</div>
                <div class="metric-value">{score:.1f}%</div>
                <div class="risk-indicator {risk_class}">
                    <span>●</span> {risk}
                </div>
                <div class="metric-desc">该分数经过CNKI校准，接近知网AIGC检测的AI特征值</div>
            </div>
            
            <div class="metric-card warning">
                <div class="metric-label">AI疑似段落占比</div>
                <div class="metric-value warning">{ratio:.1f}%</div>
                <div class="metric-desc">{ai_count} / {total} 个段落被标记为疑似AI生成</div>
            </div>
            
            <div class="metric-card success">
                <div class="metric-label">原始检测分数</div>
                <div class="metric-value success">{raw_score:.1f}%</div>
                <div class="metric-desc">未校准前的模型原始输出 (供参考)</div>
            </div>
        </div>"""
    
    def _render_distribution(self) -> str:
        """渲染分布图表"""
        paragraphs = self.data.get('paragraph_results', [])
        total = len(paragraphs)
        
        if total == 0:
            return ""
        
        # 计算前中后三部分的AI特征值
        third = max(1, total // 3)
        front = paragraphs[:third]
        middle = paragraphs[third:2*third]
        back = paragraphs[2*third:]
        
        def calc_score(paras):
            if not paras:
                return 0
            return sum(p['calibrated_score'] for p in paras) / len(paras) * 100
        
        front_score = calc_score(front)
        middle_score = calc_score(middle)
        back_score = calc_score(back)
        
        # 计算AI/人工分布
        ai_count = sum(1 for p in paragraphs if p['is_ai'])
        human_count = len(paragraphs) - ai_count
        
        return f"""
        <section class="section">
            <h2 class="section-title">
                <div class="section-icon">📊</div>
                AIGC片段分布分析
            </h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 32px;">
                <div>
                    <h3 style="font-size: 16px; margin-bottom: 16px; color: var(--text-secondary);">文档结构AI分布</h3>
                    <div class="chart-container">
                        <canvas id="distributionChart"></canvas>
                    </div>
                </div>
                <div>
                    <h3 style="font-size: 16px; margin-bottom: 16px; color: var(--text-secondary);">AI/人工段落占比</h3>
                    <div class="chart-container">
                        <canvas id="pieChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 32px;">
                <div style="text-align: center; padding: 24px; background: var(--bg-paper); border-radius: 12px;">
                    <div style="font-size: 32px; font-weight: 700; color: var(--primary);">{front_score:.1f}%</div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;">前部 {len(front)} 段</div>
                </div>
                <div style="text-align: center; padding: 24px; background: var(--bg-paper); border-radius: 12px;">
                    <div style="font-size: 32px; font-weight: 700; color: var(--primary);">{middle_score:.1f}%</div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;">中部 {len(middle)} 段</div>
                </div>
                <div style="text-align: center; padding: 24px; background: var(--bg-paper); border-radius: 12px;">
                    <div style="font-size: 32px; font-weight: 700; color: var(--primary);">{back_score:.1f}%</div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 8px;">后部 {len(back)} 段</div>
                </div>
            </div>
            
            <div class="info-box">
                <h4>📋 分布说明</h4>
                <p>文档分为前、中、后三部分，分别计算各部分的平均AIGC分数。不同部分的AI痕迹分布可以反映写作模式：
                前部通常包含摘要和引言，中部为正文主体，后部为结论和参考文献。
                各部分差异过大可能表明写作风格不统一。</p>
            </div>
        </section>"""
    
    def _render_paragraph_details(self) -> str:
        """渲染段落详情表格"""
        paragraphs = self.data.get('paragraph_results', [])
        
        rows = []
        for p in paragraphs[:30]:  # 只显示前30段
            para_id = p['para_id'] + 1
            is_ai = p['is_ai']
            score = p['calibrated_score'] * 100
            raw = p['raw_probability'] * 100
            model = p['triggered_by']
            
            # 确定分数等级
            if score < 15:
                score_class = 'low'
                level = '低'
            elif score < 30:
                score_class = 'medium'
                level = '中'
            else:
                score_class = 'high'
                level = '高'
            
            rows.append(f"""
                <tr>
                    <td>第 {para_id} 段</td>
                    <td><span class="score-badge {score_class}">{score:.1f}% ({level})</span></td>
                    <td>{raw:.1f}%</td>
                    <td><span class="model-tag">{model}</span></td>
                    <td>{'⚠️ AI疑似' if is_ai else '✓ 人工'}</td>
                </tr>""")
        
        return f"""
        <section class="section">
            <h2 class="section-title">
                <div class="section-icon">📝</div>
                分段检测结果
            </h2>
            
            <table class="data-table">
                <thead>
                    <tr>
                        <th>段落编号</th>
                        <th>CNKI校准分数</th>
                        <th>原始概率</th>
                        <th>触发模型</th>
                        <th>判定结果</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
            
            {'<p style="text-align: center; color: var(--text-secondary); margin-top: 16px; font-size: 14px;">... 仅显示前30段，完整数据请查看JSON报告</p>' if len(paragraphs) > 30 else ''}
        </section>"""
    
    def _render_suggestions(self) -> str:
        """渲染建议部分"""
        score = self.data.get('doc_calibrated_score', 0) * 100
        
        if score < 15:
            risk_desc = "低风险"
            suggestions = [
                ("整体评估良好", "文档AIGC分数在正常范围内，未发现明显的AI生成痕迹。"),
                ("继续保持", "建议保持当前的写作风格，注重个人观点和原创性表达。"),
                ("规范引用", "如使用AI辅助工具，请确保按照学术规范进行标注。")
            ]
        elif score < 30:
            risk_desc = "中风险"
            suggestions = [
                ("重点复核", "建议对标记为AI疑似的段落进行人工审查，特别关注AIGC分数>20%的段落。"),
                ("增加原创性", "在疑似AI段落中增加个人观点、案例分析、实地调研数据等原创内容。"),
                ("改写优化", "对模板化表达进行改写，使用更自然的语言风格，避免过于规范的学术套话。"),
                ("引用标注", "如确实使用AI辅助生成内容，请按照学校要求明确标注。")
            ]
        elif score < 50:
            risk_desc = "较高风险"
            suggestions = [
                ("深度审查", "文档存在明显的AI生成痕迹，建议进行全面审查和改写。"),
                ("重构内容", "对高疑似段落进行深度改写，增加个人研究思考和原创分析。"),
                ("数据支撑", "补充原始调研数据、访谈记录、实验结果等一手资料。"),
                ("学术规范", "咨询导师关于AI使用的学术规范，必要时重新撰写部分章节。"),
                ("二次检测", "修改完成后建议再次使用本系统或知网官方检测进行验证。")
            ]
        else:
            risk_desc = "高风险"
            suggestions = [
                ("紧急处理", "文档高度疑似AI生成，建议暂停提交，进行全面重写。"),
                ("重新撰写", "建议基于个人研究重新撰写论文，确保学术诚信。"),
                ("导师沟通", "立即与导师沟通，说明情况并寻求指导。"),
                ("学术诚信", "了解学校关于AI使用的具体规定，避免学术不端风险。"),
                ("官方检测", "在修改完成前，建议先使用知网官方AIGC检测进行验证。")
            ]
        
        suggestion_items = ''.join([
            f"""
            <li>
                <div class="suggestion-num">{i+1}</div>
                <div class="suggestion-content">
                    <div class="suggestion-title">{title}</div>
                    <div class="suggestion-desc">{desc}</div>
                </div>
            </li>"""
            for i, (title, desc) in enumerate(suggestions)
        ])
        
        return f"""
        <section class="section">
            <h2 class="section-title">
                <div class="section-icon">💡</div>
                修改建议 ({risk_desc})
            </h2>
            
            <ul class="suggestion-list">
                {suggestion_items}
            </ul>
        </section>"""
    
    def _render_footer(self) -> str:
        """渲染页脚"""
        return f"""
        <footer class="report-footer">
            <p><strong>免责声明</strong></p>
            <p>本系统输出仅供参考，不构成对学术不端行为的直接认定，也不替代学校官方检测系统或CNKI等权威检测结果。</p>
            <p style="margin-top: 16px;">MBA-AIGC-Detector v1.0 | 5模型并联融合检测 | CNKI校准版</p>
            <p style="margin-top: 8px; opacity: 0.7;">报告生成时间: {self.report_time}</p>
        </footer>"""
    
    def _render_charts_js(self) -> str:
        """渲染图表JavaScript"""
        paragraphs = self.data.get('paragraph_results', [])
        total = len(paragraphs)
        
        if total == 0:
            return ""
        
        # 计算分布数据
        third = max(1, total // 3)
        front = paragraphs[:third]
        middle = paragraphs[third:2*third] if len(paragraphs) > third else []
        back = paragraphs[2*third:] if len(paragraphs) > 2*third else []
        
        def calc_score(paras):
            if not paras:
                return 0
            return round(sum(p['calibrated_score'] for p in paras) / len(paras) * 100, 1)
        
        front_score = calc_score(front)
        middle_score = calc_score(middle)
        back_score = calc_score(back)
        
        ai_count = sum(1 for p in paragraphs if p['is_ai'])
        human_count = len(paragraphs) - ai_count
        
        return f"""
        // 分布柱状图
        const ctx1 = document.getElementById('distributionChart').getContext('2d');
        new Chart(ctx1, {{
            type: 'bar',
            data: {{
                labels: ['前部 ({len(front)}段)', '中部 ({len(middle)}段)', '后部 ({len(back)}段)'],
                datasets: [{{
                    label: 'AIGC分数 (%)',
                    data: [{front_score}, {middle_score}, {back_score}],
                    backgroundColor: [
                        'rgba(26, 54, 93, 0.8)',
                        'rgba(44, 82, 130, 0.8)',
                        'rgba(66, 112, 170, 0.8)'
                    ],
                    borderColor: [
                        '#1a365d',
                        '#2c5282',
                        '#4270aa'
                    ],
                    borderWidth: 2,
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return 'AIGC分数: ' + context.parsed.y + '%';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        grid: {{
                            color: 'rgba(0, 0, 0, 0.05)'
                        }},
                        ticks: {{
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }}
                    }},
                    x: {{
                        grid: {{
                            display: false
                        }}
                    }}
                }}
            }}
        }});
        
        // 饼图
        const ctx2 = document.getElementById('pieChart').getContext('2d');
        new Chart(ctx2, {{
            type: 'doughnut',
            data: {{
                labels: ['AI疑似段落', '人工疑似段落'],
                datasets: [{{
                    data: [{ai_count}, {human_count}],
                    backgroundColor: [
                        '#c53030',
                        '#276749'
                    ],
                    borderColor: [
                        '#991b1b',
                        '#1a4d2e'
                    ],
                    borderWidth: 2,
                    hoverOffset: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{
                            padding: 20,
                            font: {{
                                size: 14
                            }}
                        }}
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const total = {ai_count} + {human_count};
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return context.label + ': ' + context.parsed + '段 (' + percentage + '%)';
                            }}
                        }}
                    }}
                }}
            }}
        }});
        """


def main():
    parser = argparse.ArgumentParser(description='生成HTML格式AIGC检测报告')
    parser.add_argument('--json', type=str, required=True, help='检测结果JSON文件路径')
    parser.add_argument('--docx', type=str, required=True, help='原始DOCX文件路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出HTML文件路径')
    
    args = parser.parse_args()
    
    # 读取检测结果
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 生成报告
    generator = HTMLReportGenerator(data, args.docx)
    generator.generate(args.output)


if __name__ == '__main__':
    main()
