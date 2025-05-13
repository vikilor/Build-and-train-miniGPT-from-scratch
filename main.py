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
from config import GPTConfig
from model.gpt import GPT
from dataset import MyDataset
torch.manual_seed(1024)

print(torch.cuda.is_available())  # 如果返回 True，表示 GPU 可用
print(torch.cuda.device_count())  # 输出 GPU 的数量

#-------训练函数------使用混合精度训练，通过在训练过程中使用半精度浮点数（FP16）而非单精度浮点数（FP32）来减少内存消耗和计算时间，同时保持模型的精度。

def train(model, optimizer, scheduler, train_loader, val_loader, device, epoch):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    start_time = time.time() #记录训练时间
    for batch_idx, (x, y) in enumerate(train_loader):
        # 将数据移到设备上
        x, y = x.to(device), y.to(device)
        #x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast():  # 使用autocast进行混合精度训练
            logits, loss = model(x, targets=y, loss_mask=None)

        # 使用Scaler进行反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()  # 更新Scaler
        
        scheduler.step()
        
        total_loss += loss.item()  #loss.item()为了拿出标量值，因为这里使用了梯度，loss实际上是这样的：比如tensor(2.5000, grad_fn=<AddBackward0>)
        
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    # ------- 分布式同步 total_loss -------
    total_loss_tensor = torch.tensor(total_loss, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = total_loss_tensor.item() / dist.get_world_size()
    else:
        total_loss = total_loss_tensor.item()
        
    end_time = time.time()
    epoch_time = end_time - start_time
       
    return total_loss, epoch_time
#-------训练函数------普通训练
'''def train(model, optimizer, scheduler, train_loader, val_loader, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):  # batch_idx：第几个batch，
                                                        #x, y：就是mydataset.getitem中返回的x和y。
        # 将数据移到设备上
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        logits, loss = model(x, targets=y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 调整学习率
        scheduler.step()
        
        total_loss += loss.item() #loss.item()为了拿出标量值，因为这里使用了梯度，loss实际上是这样的：比如tensor(2.5000, grad_fn=<AddBackward0>)
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    return total_loss'''

def eval(model, val_loader, device):
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y, loss_mask=None)
            val_loss += loss.item()
          
    val_loss_tensor = torch.tensor(val_loss, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / dist.get_world_size()
    else:
        val_loss = val_loss_tensor.item()
    return val_loss

#-------训练模型-------(集中训练）
# train data
'''def main():
    train_dataset = MyDataset('./mobvoi_seq_monkey_text_gen_open_corpus.jsonl')  #实例化类

    # split traindataset to train and val
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)  #实例化
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

    model = GPT(GPTConfig(),device=device)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 打印模型一共有多少参数

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  #前11个epoch用的是3e-4
    # 设置 cosine 学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    best_val_loss = float('inf')

    start_epoch = 0
    checkpoint_path = 'checkpoints/model_best.pt' #方便继续训练
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
    else:
        print("No checkpoint found, starting at epoch0.")    

    all_start_time = time.time()  #记录开始训练时间
    for epoch in range(start_epoch, 100):
        train_loss,epoch_time = train(model, optimizer, scheduler, train_loader, val_loader, device, epoch)
        val_loss = eval(model, val_loader, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, training time: {epoch_time:.2f} seconds')

        # 保存模型
        avg_val_loss = val_loss / len(val_loader)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }
        # 保存每个epoch的模型
        os.makedirs('checkpoints', exist_ok=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, f'checkpoints/model_best.pt')
            print(f'Best model saved at epoch {epoch}')

    all_end_time = time.time()  #记录训练结束时间
    total_training_time = all_end_time - all_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    print(f"Training finished in {hours} hours {minutes} minutes {seconds} seconds.")
    
if __name__ == "__main__":
    main()    '''
#-------分布式训练--------spawn版
'''def setup(rank, world_size):
    # 初始化分布式环境
    os.environ["MASTER_ADDR"] = "10.21.181.2"  # 或者指定主节点的IP地址
    os.environ["MASTER_PORT"] = "12345"  # 选择一个可用的端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)

    # 加载数据
    train_dataset = MyDataset('./mobvoi_seq_monkey_text_gen_open_corpus.jsonl')
    train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

    # 使用 DistributedSampler 来保证每个进程获取不同的数据
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # 使用 sampler 进行数据加载
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler)

    # 创建模型
    model = GPT(GPTConfig())
    device = torch.device("cuda", rank)
    model.to(device)

    # 使用 DistributedDataParallel 包装模型
    model = DDP(model, device_ids=[rank])

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:  # 只在主进程打印
        print(f"Total parameters: {total_params / 1e6} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    best_val_loss = float('inf')
    # 训练循环
    for epoch in range(100):
        # 更新 DistributedSampler 的 epoch
        train_sampler.set_epoch(epoch)

        # 训练
        train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device, epoch)
        
        # 验证
        val_loss = eval(model, val_loader, device)

        if rank == 0:  # 只在主进程中打印输出
            print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

        # 保存模型只在主进程中进行
        avg_val_loss = val_loss / len(val_loader)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
        }

        # 保存模型
        os.makedirs('checkpoints_spawn', exist_ok=True)
        if avg_val_loss < best_val_loss and rank == 0:  # 只有主进程保存模型
            best_val_loss = avg_val_loss
            torch.save(checkpoint, f'checkpoints_spawn/model_best.pt')
            print(f'Best model saved at epoch {epoch}')

    cleanup()

if __name__ == "__main__":
    world_size = 4  #使用4块gpu
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)'''

#-------分布式训练-----torchrun版
def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def main():
    setup()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    # 加载数据
    train_dataset = MyDataset('./belle_general_qa.jsonl')
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=12, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=12, sampler=val_sampler)

    # 创建模型
    model = GPT(GPTConfig(), device=device)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Total parameters: {total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    best_val_loss = float('inf')
    
    start_epoch = 0
    checkpoint_path = 'checkpoints_torchrun/model_best.pt' #方便继续训练
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
    else:
        print("No checkpoint found, starting at epoch0.")

    for epoch in range(start_epoch,500):
        train_sampler.set_epoch(epoch)

        # 调用你的train和eval函数
        train_loss, epoch_time = train(model, optimizer, scheduler, train_loader, val_loader, device, epoch)
        val_loss = eval(model, val_loader, device)

        if rank == 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f},training time: {epoch_time:.2f} seconds')

            # 保存最优模型
            avg_val_loss = val_loss / len(val_loader)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # 注意: model.module
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }

            os.makedirs('checkpoints_torchrun', exist_ok=True)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint, f'checkpoints_torchrun/model_best.pt')
                print(f'Best model saved at epoch {epoch}')

    cleanup()

if __name__ == "__main__":
    main()







