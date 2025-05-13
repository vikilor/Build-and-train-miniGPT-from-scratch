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

#----定义参数-----
@dataclass   #统一管理参数
class GPTConfig:
    block_size = 512  #即输入的最大长度token
    batch_size = 12
    n_layer = 12  #有12个block
    n_head = 12  #多头注意力，定义有多少个头
    n_embd = 768 #隐藏层大小
    hidden_dim = 768
    dropout = 0.2
    head_size = n_embd // n_head #头的输入大小
    vocab_size = 50257 #token的数量