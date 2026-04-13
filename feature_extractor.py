#!/usr/bin/env python3
"""
特征提取器 - BERT + 统计特征
"""

import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict


class FeatureExtractor:
    """BERT + 统计特征提取器"""
    
    def __init__(self, model_name: str = 'bert-base-chinese'):
        """
        初始化特征提取器
        
        Args:
            model_name: BERT模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'
    
    def extract_bert_features(self, text: str) -> np.ndarray:
        """
        提取BERT特征（768维）
        
        Args:
            text: 输入文本
            
        Returns:
            768维特征向量
        """
        # 截断文本
        text = text[:512]
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True, 
            padding=True
        )
        
        if self.device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]token的表示
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return features[0]  # 768维
    
    def extract_stat_features(self, text: str) -> np.ndarray:
        """
        提取统计特征（13维）
        
        Args:
            text: 输入文本
            
        Returns:
            13维统计特征
        """
        char_count = len(text)
        
        # 句子分割
        sentences = re.split(r'[。！？\.\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            sentence_lengths = [len(s) for s in sentences]
            avg_sentence_length = np.mean(sentence_lengths)
            sentence_length_std = np.std(sentence_lengths)
        else:
            avg_sentence_length = sentence_length_std = 0
        
        # 标点比例
        comma_count = text.count('，') + text.count(',')
        period_count = text.count('。') + text.count('.')
        comma_ratio = comma_count / char_count if char_count > 0 else 0
        period_ratio = period_count / char_count if char_count > 0 else 0
        
        # 代词比例
        pronouns = ['我', '你', '他', '她', '它', '我们', '你们', '他们']
        pronoun_count = sum(text.count(p) for p in pronouns)
        pronoun_ratio = pronoun_count / char_count if char_count > 0 else 0
        
        # 连词比例
        conjunctions = ['和', '与', '或', '但是', '然而', '因此', '因为', '所以']
        conjunction_count = sum(text.count(c) for c in conjunctions)
        conjunction_ratio = conjunction_count / char_count if char_count > 0 else 0
        
        # 词汇多样性
        unique_chars = len(set(text))
        unique_word_ratio = unique_chars / char_count if char_count > 0 else 0
        
        # 平均词长
        words = text.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # 数字比例
        digit_count = sum(c.isdigit() for c in text)
        digit_ratio = digit_count / char_count if char_count > 0 else 0
        
        # 中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        chinese_char_ratio = chinese_chars / char_count if char_count > 0 else 0
        
        # 段落长度
        paragraph_length = char_count
        
        # 突发性（Burstiness）
        if sentences and avg_sentence_length > 0:
            burstiness = sentence_length_std / avg_sentence_length
        else:
            burstiness = 0
        
        # 正式度
        formal_words = ['研究', '分析', '策略', '管理', '企业', '市场', '发展']
        formality_score = sum(text.count(w) for w in formal_words) / char_count if char_count > 0 else 0
        
        return np.array([
            avg_sentence_length,
            sentence_length_std,
            comma_ratio,
            period_ratio,
            pronoun_ratio,
            conjunction_ratio,
            unique_word_ratio,
            avg_word_length,
            digit_ratio,
            chinese_char_ratio,
            paragraph_length,
            burstiness,
            formality_score
        ])
    
    def extract(self, text: str) -> Dict:
        """
        提取所有特征
        
        Args:
            text: 输入文本
            
        Returns:
            包含bert、stat、combined特征的字典
        """
        bert_feat = self.extract_bert_features(text)
        stat_feat = self.extract_stat_features(text)
        combined = np.concatenate([stat_feat, bert_feat])
        
        return {
            "bert_features": bert_feat,
            "stat_features": stat_feat,
            "combined_features": combined
        }
