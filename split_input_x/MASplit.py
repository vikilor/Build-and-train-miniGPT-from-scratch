class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建多个单头注意力
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # 将多头的输出投影回原始维度
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 硬性分割 x, 假设 x 的形状为 (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # 将 seq_len 切分为多个块 (num_heads 是切块的数量)
        x_chunks = torch.chunk(x, len(self.heads), dim=1)  #将x切分成多个块，x的形状是：(batch_size,seq_len,hidden_size)

        # 对每个块应用一个 SingleHeadAttention
        outputs = []
        for i, chunk in enumerate(x_chunks):
            out = self.heads[i](chunk)  # 每个头对对应块进行计算
            outputs.append(out)

        # 将所有头的输出拼接起来
        output = torch.cat(outputs, dim=-1)  # 拼接成一个大 tensor

        # 投影和 Dropout
        output = self.proj(output)
        output = self.dropout(output)

        return output
