# 从零开始搭建并训练GPT-2

## 📁 项目结构
```
GPT/
├── checkpoints
├── split_input_x         <----- 理论上的 MultiHeadAttend
├── model/
│   ├── __init__.py
│   ├── attention.py
│   ├── feedforward.py
│   ├── block.py
│   ├── gpt.py
├── dataset.py
├── config.py
├── main.py
├── process_Belle.py      <----- 处理 Belle 数据集文件
```

## 关于数据
本项目使用了序列猴子对问答和Belle数据集

###序列猴子下载链接：https://github.com/mobvoi/seq-monkey-data/blob/main/docs/ft_open_corpus.md

###belle数据集下载链接：https://huggingface.co/datasets/BelleGroup/train_1M_CN

belle数据集需要通过process_Belle.py处理。

参与训练的样本格式均为：{"input": "<提问>", "target": "<回答>", "task": "<任务类别>"}
## 关于训练模型
包含混合精度训练与普通训练，集中训练和torchrun分布式训练。
如使用分布式训练，请输入：torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 model.py
参数请根据实际调整

##****目前正在逐步优化，持续更新中********
