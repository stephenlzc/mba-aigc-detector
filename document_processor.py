#!/usr/bin/env python3
"""
文档处理器 - 读取PDF/Word并切分段落
"""

import re
import zipfile
from pathlib import Path
from typing import List


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, min_para_length: int = 100, max_para_length: int = 1000):
        """
        初始化处理器
        
        Args:
            min_para_length: 最小段落长度
            max_para_length: 最大段落长度
        """
        self.min_para_length = min_para_length
        self.max_para_length = max_para_length
    
    def read_pdf(self, path: Path) -> str:
        """
        从PDF读取文本
        
        Args:
            path: PDF文件路径
            
        Returns:
            文本内容
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("请安装pypdf: pip install pypdf")
        
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    
    def read_docx(self, path: Path) -> str:
        """
        从Word读取文本
        
        Args:
            path: Word文件路径
            
        Returns:
            文本内容
        """
        try:
            import docx
            doc = docx.Document(str(path))
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            # 备用方案：直接读取xml
            with zipfile.ZipFile(path) as archive:
                raw = archive.read("word/document.xml").decode("utf-8", errors="ignore")
            return re.sub(r"<[^>]+>", "", raw)
    
    def read_txt(self, path: Path) -> str:
        """
        从文本文件读取
        
        Args:
            path: 文本文件路径
            
        Returns:
            文本内容
        """
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def read_document(self, path: Path) -> str:
        """
        读取文档（自动识别格式）
        
        Args:
            path: 文档路径
            
        Returns:
            文本内容
        """
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.read_pdf(path)
        elif suffix == '.docx':
            return self.read_docx(path)
        elif suffix in ['.txt', '.md']:
            return self.read_txt(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    def split_paragraphs(self, text: str) -> List[str]:
        """
        切分段落
        
        Args:
            text: 输入文本
            
        Returns:
            段落列表
        """
        # 首先尝试按空行分割
        raw_paras = re.split(r'\n\s*\n', text)
        
        paragraphs = []
        for para in raw_paras:
            para = para.strip()
            para = re.sub(r'\s+', ' ', para)
            if self.min_para_length <= len(para) <= self.max_para_length:
                paragraphs.append(para)
        
        # 如果段落太少，按句子切分
        if len(paragraphs) < 5:
            sentences = re.split(r'(?<=[。！？])', text)
            current_para = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                current_para += sent
                if len(current_para) >= 200:
                    if len(current_para) <= self.max_para_length:
                        paragraphs.append(current_para)
                    current_para = ""
            if current_para and len(current_para) >= self.min_para_length:
                paragraphs.append(current_para[:self.max_para_length])
        
        return paragraphs
    
    def process(self, path: Path) -> List[str]:
        """
        处理文档：读取并切分段落
        
        Args:
            path: 文档路径
            
        Returns:
            段落列表
        """
        text = self.read_document(path)
        return self.split_paragraphs(text)
