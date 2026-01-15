from train_utils import *
import numpy as np
import torch
import os
from typing import BinaryIO,IO
import argparse
import torch.nn.functional as F
from transformer import TransformerLM
from train_utils import Adamw
import wandb
import tqdm 
from pathlib import Path
import numpy.typing as npt
# 加载以内存映射格式存储的 token id 序列
# .bin/.dat 格式 
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
    
    os.makedirs(args.save_path, exist_ok=True)

    # 2. 加载数据集
    train_data = get_memmap_dataset(args.train_data_path)
    val_data   = get_memmap_dataset(args.val_data_path)

    print(f"训练集大小: {len(train_data)} tokens")
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
    ).to(args.device)
 

    # 4. 初始化优化器
    optimizer = Adamw(model.parameters(), lr=args.lr, weight_decay=0.1)

    # 【新增】初始化梯度缩放器 (GradScaler)
    # 注意：BF16 理论上不需要 Scaler，但为了代码兼容性（如切换到 FP16）建议加上
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == 'cuda' and args.dtype == 'float16'))

    # 5. 检查点恢复逻辑
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = run_load_checkpoint(ckpt_path, model, optimizer)
        #print(f"Resuming from iteration {start_iter}")
    
    

    # 6. 初始化 WandB 
    wandb.init(
        project="cs336-assignment1",
        entity="my-cs336-work",
        name="d-model1", 
        config=args
    )

    # 7. 主训练循环
    pbar =tqdm.tqdm(range(start_iter, args.max_iters), desc="Training", leave=False)

    for it in pbar:

        # A. 更新学习率
        lr = run_get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. 训练步
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # 【核心修改 1】：使用 autocast 开启混合精度
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        # logits = model(x)
        # loss = F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
        #                         y.reshape(-1))
        
        optimizer.zero_grad()
        # loss.backward()
        # 【核心修改2】
        # 如果是 bf16，scaler.scale(loss) 实际上等同于直接 loss
        scaler.scale(loss).backward()
        
        # 【核心修改 3】：梯度裁剪前需要先 unscale
        scaler.unscale_(optimizer)
        # 梯度裁剪s
        run_gradient_clipping(model.parameters(), args.max_norm)
        
        # optimizer.step()
        # 【核心修改 4】：使用 scaler 执行优化器步进
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix(loss=loss.item(), lr=lr)
    

        # C. 验证与日志记录
        if it % 200 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = F.cross_entropy(v_logits.reshape(-1,v_logits.shape[-1])
                                         , vy.reshape(-1))
                
                #print(f"Iter {it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}, lr {lr:.2e}")
                wandb.log({
                    "train/loss": loss.item(), 
                    "val/loss": v_loss.item(), 
                    "lr": lr, 
                    "iter": it + 1
                })

        # D. 保存检查点 (每 1000 步保存一次)
        if it % 1000 == 0 and it > 0:
            hist_path = os.path.join(args.save_path, f"ckpt_{it}.pth")
            run_save_checkpoint(model, optimizer, it, hist_path)
            run_save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    # 训练结束保存最终模型
    run_save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "model.pt"))
    wandb.finish()


base_dir = Path(__file__).resolve().parent.parent  # 根目录
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default=base_dir/"checkpoints")
    parser.add_argument("--out_dir", type=str, default=base_dir/"checkpoints")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1365) # 比如 8/3 * 512 的近似
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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