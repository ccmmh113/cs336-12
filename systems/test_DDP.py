import numpy as np
import torch
import os
from typing import BinaryIO,IO
import argparse
import torch.nn.functional as F
from cs336_basics.transformer import TransformerLM
from cs336_basics.train_utils import Adamw,run_get_lr_cosine_schedule,run_gradient_clipping
import wandb
import tqdm 
from pathlib import Path
import numpy.typing as npt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
def get_memmap_dataset(path, dtype=np.int32):
    arr = np.memmap(path, dtype=dtype, mode="r")  
    return arr

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))

    x = torch.stack([
            torch.from_numpy((dataset[i : i + context_length]).astype(np.int64))
            for i in starting_idxs
    ])  
    y = torch.stack(
        [
            torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64))
            for i in starting_idxs
        ]
    )  
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }   

    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
   
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            checkpoint = torch.load(f, weights_only=False)
    else:
        checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']    


def train(args):
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)           # 这一步很重要，设定当前进程的默认显卡
    global_rank = int(os.environ["RANK"])       # 全局 rank，用于判断是否是主进程 (0)
    world_size = int(os.environ["WORLD_SIZE"])  # 显卡总数
    current_device = f"cuda:{local_rank}"
    torch.manual_seed(1337 + global_rank)
    prof = None
    if global_rank == 0:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # wait: 跳过前10步(等模型稳定), warmup: 预热5步, active: 正式记录5步
            schedule=torch.profiler.schedule(wait=10, warmup=5, active=5, repeat=1),
            on_trace_ready=tensorboard_trace_handler("./log/transformer_prof"),
            record_shapes=True,
            profile_memory=True, # 开启显存占用量化
            with_stack=True      # 关联 Python 代码行
        )
        prof.start()
    # 2. 加载数据集
    full_train_data = get_memmap_dataset(args.train_data_path)
    val_data   = get_memmap_dataset(args.val_data_path)

    chunk_size=len(full_train_data) //world_size
    train_data=full_train_data[global_rank *chunk_size :(global_rank+1)*chunk_size]
    if global_rank == 0:
        print(f"总训练集大小: {len(full_train_data)} tokens")
        print(f"每张卡分配数据量: {len(train_data)} tokens")
        print(f"验证集大小: {len(val_data)} tokens")


    # 3. 初始化模型
    model = TransformerLM(
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                d_model=args.d_model,
                num_layers=args.num_layers,
                n_head=args.n_head,
                d_ff=args.d_ff,
                rope_theta=args.rope_theta,
                device=None
        # 传入实验参数
    ).to(local_rank)
 

    # 4. 初始化优化器
    # 【新增】初始化梯度缩放器 (GradScaler)
    # 注意：BF16 理论上不需要 Scaler，但为了代码兼容性（如切换到 FP16）建议加上
    optimizer = Adamw(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # 5. 检查点恢复逻辑
    start_iter = 0
    ckpt_path = os.path.join(args.save_path, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = run_load_checkpoint(ckpt_path, model, optimizer)    

    model=DDP(model,device_ids=[local_rank])

    if global_rank == 0:
        wandb.init(
            project="cs336-assignment1",
            entity="my-cs336-work",
            name="test_DDP", 
            config=args
        )
        pbar = tqdm.tqdm(range(start_iter, args.max_iters), desc="Training", leave=False)
    else:
        pbar = range(start_iter, args.max_iters) # 其他卡不显示进度条

    # 7. 主训练循环
    for it in pbar:

        # A. 更新学习率
        lr = run_get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. 训练步
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length,current_device)

        # 使用 autocast 开启混合精度
        # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        #     logits = model(x)
        #     loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
  
        optimizer.zero_grad()

        if prof: prof.step()
        # [3] 使用 record_function 对关键块打 NVTX 标签
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with record_function("forward_pass"): # 标记前向计算
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        with record_function("backward_pass"): # 标记反向传播
            scaler.scale(loss).backward()
        # scaler.scale(loss).backward()    
        # scaler.unscale_(optimizer)
        # run_gradient_clipping(model.parameters(), args.max_norm)
        # scaler.step(optimizer)
        # scaler.update()
        with record_function("optimization_step"):
            scaler.unscale_(optimizer)
            run_gradient_clipping(model.parameters(), args.max_norm)
            scaler.step(optimizer)
            scaler.update()
        if global_rank == 0:
            pbar.set_postfix(loss=loss.item(), lr=lr)
    
        if it == 20: # 匹配上面 schedule 的总步数 (10+5+5)
            if prof:
                prof.stop()
                if global_rank == 0:
                    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # C. 验证与日志记录
        if it % 200 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, current_device)
                v_logits = model(vx)
                v_loss = F.cross_entropy(v_logits.reshape(-1,v_logits.shape[-1])
                                         , vy.reshape(-1))
                
                dist.all_reduce(v_loss, op=dist.ReduceOp.SUM)
                v_loss = v_loss / world_size

                if global_rank == 0:
                    wandb.log({
                        "train/loss": loss.item(),  # 这里的训练loss是当前卡的近似值，一般不用 reduce
                        "val/loss": v_loss.item(), 
                        "lr": lr, 
                        "iter": it + 1
                    })

        # D. 保存检查点 (每 1000 步保存一次)
        if it % 1000 == 0 and it > 0 and global_rank==0:
            run_save_checkpoint(model.module, optimizer, it, ckpt_path)
    if global_rank == 0:
    # 训练结束保存最终模型
        run_save_checkpoint(model.module, optimizer, args.max_iters, os.path.join(args.save_path, "model.pt"))
        wandb.finish()
    dist.destroy_process_group()

base_dir = Path(__file__).resolve().parent.parent  # 根目录
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default=base_dir/"checkpoints")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1365) # 比如 8/3 * 512 的近似
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_iters", type=int, default=700)
    parser.add_argument("--max_iters", type=int, default=7000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--train_data_path", type=str, default=base_dir/"data"/"TinyStoriesV2-GPT4-train.dat")
    parser.add_argument("--val_data_path", type=str, default=base_dir/"data"/"TinyStoriesV2-GPT4-valid.dat")
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    train(args)