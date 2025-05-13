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

#-----定义一个单头注意力
class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()  #要记得初始化父类！
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.head_size = config.head_size
        
        #attention_mask 通过register_buffer注册
        #因为不用计算梯度，所以节约内存和显存，速度也更快
        #tril是下三角
        self.register_buffer(
            "attention_mask",torch.tril(torch.ones(config.block_size,config.block_size))
                            )
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self,x):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)   #(batch_size,seq_len_k,head_size)
        #这里只对hidden_dim做变换，self.key是个线性变换层，nn下的线性层能批量处理，
        #所以最后变换成（batch_size, seq_len, head_size）
        q = self.query(x)  #(batch_size,seq_len_q,head_size)
        v = self.value(x)   #(batch_size,seq_len_v,head_size)
        weight = q @ k.transpose(-2,-1)  #将k的seq_len和head_size转置，
                                        #变成(batch_size, num_heads, head_size, seq_len_k)，
                                        #点积之后weight变成(batch_size, num_heads, seq_len_q, seq_len_k)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float('-inf')    #权重掩码，将attention mask中0的位置对应的weight元素，替换成负无穷
        ) / math.sqrt(self.head_size)     #当 query 和 key 的维度很大时，它们点积的结果会变得非常大，
                                            #从而使 softmax 的梯度非常小，训练变得困难（梯度消失问题）
                                            #这个公式是标准的transformer attention里的
        weight = F.softmax(weight, dim=-1)  #针对每条q中每个k做归一化
        weight = self.dropout(weight)  #随机让一些weight的值变成0
        out = weight @ v   #（seq_len_q,head_size)表示的是每个q对于所有k的值，再加权对应k的v。
        return out

#------创建多头------
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )   #这里将完整的X输入到每个单头中，但由于每个单头学到的权重不一样，所以实际单头输出的信息是不一样的。
        self.proj = nn.Linear(config.n_embd, config.n_embd) # 让不同头输出的信息做一下特征融合
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],   #真实的gpt代码是用一个大矩阵，一次性把x映射成所有heads的q/k/v，然后在计算attention的时候逻辑上切分成多个head。
            dim=-1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output    