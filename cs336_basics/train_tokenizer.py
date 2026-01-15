import os
import time
import heapq
from pretokenization_example  import find_chunk_boundaries
from collections import defaultdict
from collections import Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Dict, Tuple, Union, Pattern
import regex as re
import json
from pathlib import Path
PAT_COMPILED = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def pre_count(
        args: Tuple[str, int, int, Dict[str, int], Union[Pattern, None]]
        ) -> Counter:
        """
        每个子进程执行的任务：读取特定字节范围，解码并统计词频
        """
        file_path, start, end, special_token_to_id, delimiter_pattern = args
        with open(file_path,"rb") as f:
            f.seek(start)
            chunk_bytes=f.read(end-start)
        
        chunk=chunk_bytes.decode("utf-8", errors="ignore")
        special_tokens = set(special_token_to_id.keys())
        local_counter = Counter()
        # 按分隔符切分子块
        if delimiter_pattern:
            sub_chunks = delimiter_pattern.split(chunk)
        else:
            sub_chunks = [chunk]

        for sub_chunk in sub_chunks:

            if not sub_chunk:
                continue

            if sub_chunk in special_tokens:
                token_id = special_token_to_id[sub_chunk]
                local_counter[(token_id,)] += 1
            else:
                # 提取非空白字符并转为字节元组
                for word_str in PAT_COMPILED.findall(sub_chunk):
                    if word_str:
                        byte_sequence = tuple(word_str.encode("utf-8"))
                        local_counter[byte_sequence] += 1
        return local_counter  #{ (104, 105): 500,  (116, 104, 101): 1200,  (50256,): 15 }


"""
初始化链表和堆
将每个单词转化为一个双向链表。例如单词 fast 变为 f <-> a <-> s <-> t。
遍历所有单词，统计所有相邻对（如 (f, a), (a, s), (s, t)）及其总频次。
将这些对及频次放入优先队列中

从堆中弹出频次最高的字节对 (A, B)。
通过一个索引（通常是哈希表 pair_to_positions），直接找到语料中所有出现 (A, B) 的链表节点位置。
合并：在链表中将 A <-> B 替换为 AB
"""

"""
动态更新：

由于 A 的左边 L 失去了右邻居 A，变成了新邻居 AB，所以 (L, A) 的频次在堆中减 1，(L, AB) 的频次加 1。

同理，B 的右边 R 失去了左邻居 B，变成了新邻居 AB，所以 (B, R) 的频次减 1，(AB, R) 的频次加 1。

"""
class Node:
    __slots__ = ['value', 'word_freq', 'prev', 'next', 'active']
    """表示词内一个 token 节点，便于链表原地更新。"""
    def __init__(self, value, word_freq):
        self.value = value
        self.word_freq = word_freq  # {'count': n} 的字典
        self.prev = None
        self.next = None
        self.active=True

class BPE_TRAIN:
    def __init__(self,  input_path,vocab_size: int,special_tokens:list[str],num_chunks: int = 4,
            num_processes: int = None
            ):
            self.input_path=input_path
            self.vocab_size = vocab_size
            self.special_tokens = special_tokens
            self.num_chunks=num_chunks
            self.num_processes=num_processes
            # 初始化基础词表 (0-255)
            self.vocab = {i: bytes([i]) for i in range(256)}
            self.merges = [] # 记录合并顺序 [(p1, p2), ...]
            
            # 核心数据结构
            # self.pair_to_nodes = defaultdict(set)
            # self.pair_freqs = Counter()
            # self.pq = [] # 优先队列
            self.before_pretokenization_time = time.time()




    # 1. Set up initial byte vocab and special tokens
    def vocab_init(self):
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        #反向映射表
        self.byte_to_token_id = {v: k for k, v in self.vocab.items()}
        self.special_token_to_id = {
            token: self.byte_to_token_id[token.encode("utf-8")] for token in self.special_tokens
        }

    # 2. Prepare special tokens regex delimiter
    def rex_init(self):
        self.delimiter_pattern_compiled = None
        if self.special_tokens:
            
            special_tokens_sorted = sorted(
                [t.encode("utf-8") for t in self.special_tokens], key=len, reverse=True
            )
            escaped_tokens = [re.escape(t.decode("utf-8")) for t in special_tokens_sorted]
            delimiter_re = "|".join(escaped_tokens)
            if delimiter_re:
                self.delimiter_pattern_compiled = re.compile(f"({delimiter_re})")

    # 3. Read file, split into chunks for multiprocessing
    def multiprocessing_file(self):
        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries( f, self.num_chunks, "<|endoftext|>".encode("utf-8"))
          
        chunk_args=[]

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunk_args.append(
                (self.input_path,
                start,
                end,
                self.special_token_to_id,
                self.delimiter_pattern_compiled)
            )
            # f.seek(start)
            # chunk = f.read(end - start).decode("utf-8", errors="ignore")

        processes_to_use = self.num_processes or min(cpu_count(),8)

        processes_to_use = min(processes_to_use, len(chunk_args))

        elapsed = time.time() - self.before_pretokenization_time
        print(f"Time taken before pretokenization: {elapsed:.2f} seconds")

        self.all_word_freqs = Counter()
        start_time = time.time()
        with Pool(processes=processes_to_use) as pool:
            print(
                f"开始预处理：{processes_to_use} 处理{len(chunk_args)} 块..."
            )
            results_iterator = pool.imap_unordered(pre_count, chunk_args)
            for chunk_counter in tqdm(
                results_iterator, 
                total=len(chunk_args),
                desc="Processing chunks", 
                leave=True
            ):
                self.all_word_freqs.update(chunk_counter)

        print(f"词频处理时间: {time.time() - start_time:.2f} seconds")

    ### Pre-tokenization 结束

    def start_bpe(self):
        pair_to_nodes = defaultdict(set)
        for word_tuple, count in tqdm(self.all_word_freqs.items(), desc="Building", leave=True):

            if len(word_tuple) < 2:
                continue

            # 所有链表节点共享 word_freq 引用，节省内存
            word_freq = {'count': count}

            head = Node(word_tuple[0], word_freq)
            prev_node = head
            for i in range(1, len(word_tuple)):
                curr_node = Node(word_tuple[i], word_freq)
                prev_node.next = curr_node
                curr_node.prev = prev_node

                pair = (prev_node.value, curr_node.value)
                pair_to_nodes[pair].add(prev_node)
                prev_node = curr_node


        pair_freqs = Counter()
        for pair, nodes in tqdm(pair_to_nodes.items(), desc="Counting pairs", leave=True):
            # 某个 pair 的出现次数 = 其所有节点所对应 word 的词频累加
            pair_freqs[pair] = sum(node.word_freq['count'] for node in nodes)

        pq = []
        for p ,f in pair_freqs.items():
            if f>0:
                byte_pair=self.vocab[p[0]]+self.vocab[p[1]]
                pq.append(
                    (-f,byte_pair,p)
                )
                
        heapq.heapify(pq)


        ### BPE 开始
        num_merges = self.vocab_size - len(self.vocab)
        pbar = tqdm(total=num_merges, desc="bpe合并")
        start_time = time.time()

        for _ in range(num_merges):
            if not pq:
                break

            # 取出频率最高的 pair，处理优先队列惰性删除的过期元素
            best_pair = None
            while pq:
                freq,_,pair = heapq.heappop(pq)
                freq=-freq

                if pair not in pair_freqs:
                    continue  # 已经被合并删除
                if pair_freqs[pair] == freq:
                    best_pair = pair
                    break

            if best_pair is None:
                break

            p1, p2 = best_pair

            # 合成新 token，添加到 merges/vocab
            new_token_id = len(self.vocab)
            merged_token_bytes = self.vocab[p1] +self.vocab[p2]
            self.merges.append((self.vocab[p1], self.vocab[p2]))
            self.vocab[new_token_id] = merged_token_bytes

            # 逐个更新包含改 pair 的词
            nodes_to_process = list(pair_to_nodes[best_pair])
            for node1 in nodes_to_process:
                # [修复 3] 跳过已经被其他合并操作注销的“僵尸节点”
                if not node1.active:
                    continue
                node2 = node1.next
                # 确保右节点有效，且值匹配（防止脏读）
                if node2 is None or not node2.active or node1.value != p1 or node2.value != p2:
                    continue
                word_freq = node1.word_freq['count']

                # 更新左侧相邻 pair 的频率及映射关系
                if node1.prev:
                    left = node1.prev
                    old_left_pair = (left.value, node1.value)
                    pair_freqs[old_left_pair] -= word_freq
                    heapq.heappush(pq, 
                                   (-pair_freqs[old_left_pair],
                                    
                                    self.vocab[old_left_pair[0]]+self.vocab[old_left_pair[1]]
                                    ,old_left_pair
                                    )
                                )

                    pair_to_nodes[old_left_pair].discard(left)
                    new_left_pair = (left.value, new_token_id)
                    pair_to_nodes[new_left_pair].add(left)
                    pair_freqs[new_left_pair] += word_freq
                    heapq.heappush(pq, 
                                   (-pair_freqs[new_left_pair],
                                    self.vocab[new_left_pair[0]]+ self.vocab[new_left_pair[1]],
                                    new_left_pair 
                                    )
                                )

                # 更新右侧相邻 pair 的频率及映射关系
                if node2.next:
                    right = node2.next
                    old_right_pair = (node2.value, right.value)
                    pair_freqs[old_right_pair] -= word_freq
                    heapq.heappush(pq, 
                                   (-pair_freqs[old_right_pair],  
                                    self.vocab[old_right_pair[0]]+self.vocab[old_right_pair[1]],
                                    old_right_pair)
                                    )

                    new_right_pair = (new_token_id, right.value)
                    pair_to_nodes[old_right_pair].discard(node2)
                    pair_to_nodes[new_right_pair].add(node1)
                    pair_freqs[new_right_pair] += word_freq
                    heapq.heappush(pq,(-pair_freqs[new_right_pair],
                                        self.vocab[new_right_pair[0]]+self.vocab[new_right_pair[1]]
                                        , new_right_pair
                                        
                                        ))

                # 链表合并：node1、node2合成 new_token_id
                node1.value = new_token_id
                node1.next = node2.next
                if node2.next:
                    node2.next.prev = node1
                node2.active = False

            # 删除被合并 pair 的所有统计
            del pair_freqs[best_pair]
            del pair_to_nodes[best_pair]

            pbar.update(1)

        end_time = time.time()
        print(f"Merge time: {end_time - start_time:.2f} seconds")
        pbar.close()
        return self.vocab, self.merges
    

def bytes_to_unicode():
        """
        创建一个映射，将 0-255 字节映射为一组可见的 Unicode 字符。
        这是 GPT-2 源码中的标准做法。
        """
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

def save(vocab,merges,out_dir):
        byte_encoder = bytes_to_unicode()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # 1. 词表保存：Bytes -> 可见字符串
        # 比如空格 (32) 会变成 'Ġ'，这样 JSON 看起来非常整洁
        json_vocab = {
            str(idx): "".join(byte_encoder[b] for b in b_val)
            for idx, b_val in vocab.items()
        }
        with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(json_vocab, f, indent=4, ensure_ascii=False)

        # 2. 合并规则保存：s1 s2
        with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
            
            for (s1, s2) in merges:
                # 从 vocab 中获取对应的原始字节

                s1 = "".join(byte_encoder[b] for b in s1)
                s2 = "".join(byte_encoder[b] for b in s2)
                f.write(f"{s1} {s2}\n")
            
        print(f"YESYES 模型已成功保存至目录: {out_dir}")

base_dir = Path(__file__).resolve().parent.parent  # 根目录

    
def main():
    # 文件路径配置
    input_path = base_dir / "data"/ "TinyStoriesV2-GPT4-train.txt" # 你的原始文本路径
    vocab_size = 10000 # 作业要求的词表大小
    special_tokens = ["<|endoftext|>"]
    output_dir = base_dir / "json"
    trainer = BPE_TRAIN(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
    trainer.vocab_init()
    trainer.rex_init()
    trainer.multiprocessing_file()
    vocab, merges = trainer.start_bpe()
    print("\n保存 Tokenizer...")
    # 保存
    save(vocab, merges,output_dir)
if __name__ == "__main__":
    main()