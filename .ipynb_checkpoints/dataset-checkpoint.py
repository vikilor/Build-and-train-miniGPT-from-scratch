import torch                        #导入包
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
import math
from torch.cuda.amp import autocast, GradScaler
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import time

#---------创建dataset-------
class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        import tiktoken  #gpt自己的tokenizer
        self.enc = tiktoken.get_encoding("gpt2")  #不同的gpt版本要get不一样的
        self.block_size = block_size

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}  #这个特殊符号我加在了每个问答对的后面，告诉模型这个问答对到这里就结束了
        )[0]  #为什么要[0]?是因为这个特殊符号encode之后返回的是[50256],一个单元素list，我们需要把这个50256加在问答对tokens的后面组成[4,2,5,7,3,1,...,50256].

        import json

        self.encoded_data = []

        self.max_lines = None  #如果不需要太多训练数据，可以定义一下最多读到第几行
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if self.max_lines is not None and i >= self.max_lines:
                    break
                try:
                    data = json.loads(line.strip())
                    input_text = data['input']
                    target_text = data['target']
                    task_text = data['task']
                    # 拼接成一个完整文本
                    full_text = f"问：{input_text}\n答：{target_text}"
                    #raw_data.append(full_text)
                    # 编码文本
                    encoded = self.enc.encode(full_text)
                    if len(encoded) + 1 > self.block_size + 1:
                        continue  # 跳过过长样本

                    # 添加结束符
                    encoded += [self.eos_token]

                    # padding 到 block_size + 1
                    pad_len = self.block_size + 1 - len(encoded)
                    encoded += [self.eos_token] * pad_len  # padding

                    self.encoded_data.append(encoded)
                except:
                    continue
                    
        
        '''full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])  # 将所有的问答对放成了一行
        
        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i+self.block_size+1] #上面认为地加入了<|endoftext|>这个token
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)'''
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        # 构建 loss mask：只对 “答：” 之后的部分计算 loss
        '''text = self.decode(chunk)  # 解码回文本 (这部分后面才添加的，所以编码又解码..为了验证只关注答部分会不会好一点)
        try:
            ans_start = text.index("答：")  # 找到“答：”的位置
            ans_token_start = len(self.encode(text[:ans_start + 2]))  # 包括“答：”两个字
        except ValueError:
            ans_token_start = len(chunk)  # 没找到“答：”，默认不训练任何 token

        # 构建 mask: 和 y 同长，答部分为1，其它为0
        loss_mask = torch.zeros_like(y, dtype=torch.float)
        loss_mask[ans_token_start - 1:] = 1.0  # -1是因为y向后偏移一位'''
        return x, y#, loss_mask

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)