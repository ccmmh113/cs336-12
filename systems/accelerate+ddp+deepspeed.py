import numpy as np
import torch
import os
import argparse
from torch.utils.data  import DataLoader,Dataset
import torch.nn.functional as F
from cs336_basics.transformer import TransformerLM
from cs336_basics.train_utils import run_get_lr_cosine_schedule
import wandb
import tqdm 
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import get_cosine_schedule_with_warmup
import json
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
class MyDataset(Dataset):
    def __init__(self,path,context_len,dtype=np.int32):
        self.data =np.memmap(path,dtype=dtype,mode='r')
        self.context_len=context_len
    def __len__(self):
        return len(self.data) -self.context_len
    def __getitem__(self, index):
        chunk = torch.from_numpy(
                    self.data[index:index+self.context_len+1]
                        ).long()
        x=chunk[:-1]
        y=chunk[1:]
        return x,y
def save_config(args, save_path):
    # 将 argparse 的 Namespace 对象转换为字典
    config_dict = vars(args)
    # 也可以手动清理掉一些不需要存入 json 的参数，比如路径
    config_dict = {k: v for k, v in config_dict.items() if "path" not in k}
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)


def get_data(loader):
    while True:
        for batch in loader:
            yield batch

def train(args):

    os.makedirs(args.save_path, exist_ok=True)
    mixed_precision = "no"
    if args.dtype == "float16":
        mixed_precision = "fp16"
    elif args.dtype == "bfloat16":
        mixed_precision = "bf16"

    accelerator = Accelerator(mixed_precision=mixed_precision)

    set_seed(1337 + accelerator.process_index)
    # -----------------------------
    # 1. 初始化 start_iter
    # -----------------------------
    start_iter = 0

    # -----------------------------
    # 2. 数据集
    # -----------------------------
    train_data = MyDataset(args.train_data_path, args.context_length)
    val_data = MyDataset(args.val_data_path, args.context_length)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=4
    )

    # -----------------------------
    # 3. 模型
    # -----------------------------
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_head=args.n_head,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=None
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_iters,
        num_training_steps=args.max_iters
        )
    model, optimizer, train_loader, val_loader ,scheduler= accelerator.prepare(
        model, optimizer, train_loader, val_loader,scheduler
    )

    # -----------------------------
    # 4. resume checkpoint
    # -----------------------------
    checkpoint_path = os.path.join(args.save_path, "checkpoint")

    if os.path.exists(checkpoint_path):
        accelerator.load_state(checkpoint_path)

        iteration_file = os.path.join(args.save_path, "iteration.json")
        if os.path.exists(iteration_file):
            with open(iteration_file, "r") as f:
                start_iter = json.load(f)["iteration"]

    # -----------------------------
    # 5. profiler（放到 resume 后）
    # -----------------------------
    prof = None
    if accelerator.is_main_process and args.profile and start_iter < 20:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=3, active=10, repeat=1),
            on_trace_ready=tensorboard_trace_handler("./log/transformer_prof"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

    # -----------------------------
    # 6. wandb
    # -----------------------------
    if accelerator.is_main_process:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project="cs336-assignment1",
            entity="my-cs336-work",
            name="accelerate+deepspeed",
            config=vars(args)
        )
        pbar = tqdm.tqdm(range(start_iter, args.max_iters))
    else:
        pbar = range(start_iter, args.max_iters)


    if start_iter > 0:
        train_loader = accelerator.skip_first_batches(train_loader, num_batches=start_iter)

    # 然后再初始化无限数据流
    train_iter = iter(get_data(train_loader))

    # -----------------------------
    # 7. 主训练循环
    # -----------------------------
    for it in pbar:

        # lr = run_get_lr_cosine_schedule(
        #     it,
        #     args.lr,
        #     args.min_lr,
        #     args.warmup_iters,
        #     args.max_iters
        # )

        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = lr

        model.train()
        with record_function("data_loading"):
            x, y = next(train_iter)

        optimizer.zero_grad()

        with record_function("forward_pass"):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1)
            )

        with record_function("backward_pass"):
            accelerator.backward(loss)

        with record_function("optimizer_step"):
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_norm)

        optimizer.step()
        scheduler.step()
        # logits = model(x)
        # loss = F.cross_entropy(
        #     logits.reshape(-1, logits.shape[-1]),
        #     y.reshape(-1)
        # )

        # accelerator.backward(loss)

        # if accelerator.sync_gradients:
        #     accelerator.clip_grad_norm_(model.parameters(), args.max_norm)

        if prof is not None:
            prof.step()

        if accelerator.is_main_process:
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        # -----------------------------
        # validation
        # -----------------------------
        if it % 200 == 0 or it == args.max_iters - 1:

            model.eval()

            total_loss = 0
            count = 0

            with torch.no_grad():
                for vx, vy in val_loader:
                    v_logits = model(vx)

                    v_loss = F.cross_entropy(
                        v_logits.reshape(-1, v_logits.shape[-1]),
                        vy.reshape(-1)
                    )

                    total_loss += v_loss.item()
                    count += 1

            mean_val_loss = total_loss / max(count,1)
            mean_val_loss = accelerator.reduce(
                torch.tensor(mean_val_loss, device=accelerator.device),
                reduction="mean"
            )

            if accelerator.is_main_process:
                wandb.log({
                    "train/loss": loss.item(),
                    "val/loss": mean_val_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "iter": it + 1
                })

        # -----------------------------
        # checkpoint
        # -----------------------------
        if it % 1000 == 0 and it > 0:
            accelerator.wait_for_everyone()

            accelerator.save_state(checkpoint_path)

            if accelerator.is_main_process:
                with open(os.path.join(args.save_path, "iteration.json"), "w") as f:
                    json.dump({"iteration": it + 1}, f)

        # -----------------------------
        # profiler stop
        # -----------------------------
    if  prof is not None:
            prof.stop()

    # -----------------------------
    # 最终保存
    # -----------------------------
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_path = os.path.join(args.save_path, "model")
        os.makedirs(save_path, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)

        accelerator.save_model(unwrapped_model, save_path)

        save_config(args, save_path)

        wandb.finish()
base_dir = Path(__file__).resolve().parent.parent  # 根目录
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default=base_dir/"result")
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--wandb_offline", action="store_true")
    args = parser.parse_args()
    train(args)