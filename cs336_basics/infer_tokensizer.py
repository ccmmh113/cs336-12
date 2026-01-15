import os
import json
import regex as re
import numpy as np
from train_tokenizer import bytes_to_unicode
from tqdm import tqdm
from typing import Iterable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
PAT_COMPILED = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
class INFER_TOKENIZER:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        
        self.vocab = vocab
        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        # self.merges = merges
        self.merges = dict(zip(merges, range(len(merges))))

        self.special_tokens = special_tokens
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]

    
        self.vocab_size = len(vocab)
        # 检查特殊 token 是否都在词表里，并获取它们的真实 ID
    
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.byte_to_token_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.byte_to_token_id[token_bytes] = new_id

    def encode(self, text: str) -> list[int]:
        
        tokens = []

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])
            else:
                tokens.extend(self._encode_text_segment(part))

        return tokens

    def _encode_text_segment(self, text: str) -> list[int]:
        """
        内部核心函数：对不含特殊 Token 的纯文本片段应用 BPE 合并逻辑。
        """
        ids = []
        # 例如："Hello world!" -> ["Hello", " world", "!"]
        pre_tokens = PAT_COMPILED.findall(text)
        
        for p_tok in pre_tokens:
            # 将当前片段转为字节序列，并将每个字节看作一个独立的部分”
            # 例如："Hello" -> [b'H', b'e', b'l', b'l', b'o']
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]
            
            # 第二步：反复执行合并，直到没有符合条件的合并规则为止
            while len(byte_parts) >= 2:
                # 在当前序列的所有相邻对中，寻找合并优先级最高（Rank 最小）的一对，即按照构造merge时添加pair的顺序进行合并
                best_pair = None
                min_rank = float('inf')
                
                for i in range(len(byte_parts) - 1):
                    pair = (byte_parts[i], byte_parts[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair
                
                # 如果找不到任何可以合并的规则，退出当前片段的合并过程
                if best_pair is None:
                    break 
                
                # 第三步：执行合并操作。
                # 遍历当前序列，将所有出现的 best_pair 替换成合并后的长字节块。
                new_byte_parts = []
                i = 0
                # [b'H', b'e', b'l', b'l', b'o', b'H', b'e'] -> [b'He', b'l', b'l', b'o', b'He']
                while i < len(byte_parts):
                    # 如果当前两个部分匹配最高优规则
                    if i < len(byte_parts) - 1 and (byte_parts[i], byte_parts[i+1]) == best_pair:
                        new_byte_parts.append(best_pair[0] + best_pair[1])
                        i += 2 # 跳过下一项，因为已经合并了
                    else:
                        new_byte_parts.append(byte_parts[i])
                        i += 1
                byte_parts = new_byte_parts # 更新序列，进入下一轮 while 循环
            
            # 第四步：将合并到极限后的所有字节块转换为词表中的 ID
            ids.extend(self.byte_to_token_id[part] for part in byte_parts)
                
        return ids
    def decode(self, ids: list[int]) -> str:
    
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        
        return full_bytes.decode("utf-8", errors="replace")




def load_trained( out_dir, special_tokens=None):
    vocab_path = os.path.join(out_dir, "vocab.json")
    merges_path = os.path.join(out_dir, "merges.txt")
    
    # 1. 还原字节
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}   #反向映射

    # 2. 还原词表
    with open(vocab_path, "r", encoding="utf-8") as f:
        v_raw = json.load(f)
        # 关键：Unicode 符号还原为 Bytes
        vocab = {int(k): bytes([byte_decoder[c] for c in v]) for k, v in v_raw.items()}


    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for  line in f:
            line = line.strip()
            if not line: continue
            parts=line.split()
            if len(parts) != 2: continue
            p1 = bytes([byte_decoder[c] for c in parts[0]])
            p2 = bytes([byte_decoder[c] for c in parts[1]])
            merges.append((p1,p2))
    

    tokenizer = INFER_TOKENIZER(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens
    )

    return tokenizer

def batch_tokenize(batch, tokenizer):
    out = []
    for line in batch:
        out.extend(tokenizer.encode(line))
    return np.array(out, dtype=np.int32)

def encode_txt_as_array(tokenizer, path_to_txt, save_path, batch_size=4096, n_workers=8):
    # 1.分batch
    batches = []
    with open(path_to_txt) as f:
        batch = []
        for line in f:
            batch.append(line)
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
    
    total_tokens = 0
    results = []
    # 2.多进程tokenize
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = []
        for batch in batches:
            futures.append(exe.submit(batch_tokenize, batch, tokenizer))
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing"):
            arr = fut.result()
            results.append(arr)
            total_tokens += arr.shape[0]
    
    # 3.写memmap
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokens_mm = np.memmap(save_path, dtype=np.int32, mode='w+', shape=(total_tokens,))
    pos = 0
    for arr in results:
        tokens_mm[pos:pos+arr.shape[0]] = arr
        pos += arr.shape[0]
    tokens_mm.flush()


base_dir = Path(__file__).resolve().parent.parent  # 根目录

def main():
    tokenizer = load_trained(
                                out_dir= base_dir , 
                                special_tokens=["<|endoftext|>"]     
                            )
    print("开始创建训练集。。。")
    encode_txt_as_array(
                        tokenizer, 
                        path_to_txt = base_dir / "data"/ "TinyStoriesV2-GPT4-train.txt" , 
                        save_path = base_dir / "data"/ "TinyStoriesV2-GPT4-train.dat",                                
                        batch_size=10000, 
                        n_workers=8)
    print("训练集创建完成！！！开始创建验证集")
    encode_txt_as_array(tokenizer, 
                        path_to_txt  =base_dir / "data"/ "TinyStoriesV2-GPT4-valid.txt" , 
                        save_path = base_dir / "data"/ "TinyStoriesV2-GPT4-valid.dat", 
                        batch_size=10000, 
                        n_workers=8)

    print("成功保存验证集")
if __name__ == "__main__":
    main()