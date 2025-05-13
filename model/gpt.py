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
from .block import Block
import time

#------完整的GPT MODEL-----
class GPT(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)#最后一个线性层
        
        # linear (config.n_embd -> config.vocab_size)；实际上 weight shape 是 config.vocab_size * config.n_embd，
        # 使用tie weight,为了减少embedding层学习的权重参数(可学习参数越多，模型越难训练)，加快训练
        # 使 embedding weight 和 lm_head weight 共享
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 这里使用的是正态分布初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None):  #targets是模型训练时使用的标签labels
        # idx 是输入的 token ids
        batch, seq_len = idx.size()   #idx.shape(batch_size,seq_len)
        token_emb = self.token_embedding_table(idx)   #token_emb.shape = (btach_size,seq_len,n_embd)

        # seq 长度是这次输入的最大长度
        pos_emb = self.position_embedding_table(
            # 要确保 位置编码和输入的 idx 在同一个设备上
            torch.arange(seq_len, device=idx.device)   #pos_emb.shape = (btach_size,seq_len,n_embd)
        )
        x = token_emb + pos_emb   # shape is (batch, seq_len, n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)   # shape is (batch, seq_len, vocab_size)
        
        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)   #交叉熵的输入要求：input: shape 是 [N, C]，表示有 N 个样本，每个样本有 C 个类别的预测（logits） target: shape 是 [N]，表示 N 个目标类别的索引。
            if loss_mask is not None:
                loss_mask = loss_mask.view(batch * seq_len)  # 将loss_mask调整为(batch * seq_len)的形状
                loss = F.cross_entropy(logits, targets, reduction='none')  # 计算每个token的损失
                loss = loss * loss_mask  # 只保留答部分的损失
                loss = loss.sum() / loss_mask.sum()  # 对答部分的损失求平均
            else:
                loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens,eos_token_id=50256):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:] #超过最大输入限度的话就截断
            # 获取预测
            logits, _ = self(idx_cond)
            #logits.shape = (batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :]  #取最后一个值就行了，因为最后一个值是预测输出的第一个词。这里说的值，shape=vocab_size.
            # 应用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            # 采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            # 附加到序列上
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq+1)
            if (idx_next == eos_token_id).all():  # 所有 batch 都生成了 eos（如果你用的是 batch）
                break
        return idx