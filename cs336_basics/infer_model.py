import  torch
from  transformer import  TransformerLM
from  infer_tokensizer import load_trained
from pathlib import Path
import argparse
import os
from train_utils import Adamw
from train_model import  run_load_checkpoint
def main():
    # 路径配置
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
 #—————————————————————————————————————————————— 参数配置————————————————————————————————————————
    parser.add_argument("--checkpoint_path", type=str, default=base_dir / "checkpoints"/"model.pt")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--d_ff", type=int, default=1365) # 比如 8/3 * 512 的近似
    args = parser.parse_args()

    # 1. 加载分词器
    print("正在加载分词器...")
    tokenizer = load_trained(out_dir=base_dir/"json", special_tokens=["<|endoftext|>"])  #
    eos_id = tokenizer.byte_to_token_id.get(b"<|endoftext|>", None)

    # 2. 初始化模型架构
    print("正在初始化模型结构...")
    model = TransformerLM(                       
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_head=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=None
    ).to(args.device)

    optimizer = Adamw(model.parameters(), lr=args.lr, weight_decay=0.1)
    # 3. 加载训练好的权重
    if os.path.exists(args.checkpoint_path):
        print(f"正在从 {args.checkpoint_path} 加载权重...")
        checkpoint = run_load_checkpoint(args.checkpoint_path,model,optimizer )
    else:
        print(f"错误：未找到权重文件 {args.checkpoint_path}")
        return

    model.eval()
    # 4. 交互式对话
    print("\n" + "=" * 30)
    print("模型已就绪！输入 'q' 退出。")
    print("=" * 30)

    while True:
        user_prompt = input("\nPrompt > ")
        if user_prompt.lower() in ['q', 'quit', 'exit']:
            break

        # A. 编码：文字 -> IDs
        input_ids = tokenizer.encode(user_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=args.device)

        # B. 生成：计算过程
        print("........正在思考.......", end="\r")
        with torch.no_grad():
            output_ids = model.generate(
                x= input_tensor,
                max_new_tokens=100,
                eos_token_id=eos_id,
                temperature=0.8,
                top_p=0.9
            )

            # C. 解码：IDs -> 文字
            # output_ids[0] 包含了 prompt 和生成的文本，我们全部解码
        response_text = tokenizer.decode(output_ids[0].tolist())

        print(f"Response > {response_text}")
        print("-" * 30)


if __name__ == "__main__":
    main()