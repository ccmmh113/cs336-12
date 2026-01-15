from __future__ import annotations
import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
import torch.nn.functional as F
logger = logging.getLogger(__name__)

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):

        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3*std, b=3*std),
            requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    '''
    输入：（batch，seq_len)
    输出：（batch，seq_len,dim)
    
    '''
    def __init__(self,
                 vocab_size:int,
                 d_model:int):

        super().__init__()
        std=1
        self.weight=nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(vocab_size,d_model),std=std,a=-3* std ,b=3*std
            ),
            requires_grad=True
        )
    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids, :]
    
class RoPositionalEmbedding(nn.Module):

    def __init__(self,
                d_k:int, 
                context_len:int,
                theta:float=10000.0,
                device=None
                ):
        super().__init__()
        self.d_k=d_k

        assert d_k % 2 == 0
        t = torch.arange(context_len,device=device)
        # 不容易溢出
        freqs = torch.exp(torch.arange(0, d_k, 2).float() * -(math.log(theta) / d_k))
        freqs_matrix=torch.outer(t,freqs)
        self.register_buffer("cos_cached",freqs_matrix.cos(),persistent=False)
        self.register_buffer("sin_cached",freqs_matrix.sin(),persistent=False)

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
           
            '''
              x 形状通常为:              [batch, n_heads, seq_len, d_k]
              token_position 形状通常为: [batch, seq_len] 或 [seq_len]
              
            ''' 
            cos=self.cos_cached[pos_ids].to(x.dtype)
            sin=self.sin_cached[pos_ids].to(x.dtype)
            # if x.ndim == 4:
            #     cos = cos.unsqueeze(1)
            #     sin = sin.unsqueeze(1)
            x1, x2 = rearrange(x, '... (xy half_d) -> xy ... half_d', xy=2)
            
            #x1 的形状是 (batch, n_heads, seq_len, d_k // 2)
            # 而 cos 的形状是 (batch, seq_len, d_k // 2)
            
            # 3. 旋转变换
            x_rot1 = x1 * cos - x2 * sin
            x_rot2 = x1 * sin + x2 * cos
            result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x_rot1, x_rot2).contiguous()
            # 4. 合并回去
            return result


        
    
class Causalmhattention(nn.Module):
    def __init__(self,
                 d_model,
                 n_head:int,
                 positional_encoder: RoPositionalEmbedding,
                ):
        super().__init__()

        assert d_model % n_head ==0
        self.d_model=d_model
        self.n_head= n_head
        self.d_k=d_model//n_head

        self.q_proj = Linear(d_model,d_model)
        self.k_proj = Linear(d_model,d_model)
        self.v_proj = Linear(d_model,d_model)
        self.out_proj    = Linear(d_model,d_model)
        self.positional_encoder = positional_encoder  # RoPE
   


    def forward(self, x: Float[Tensor, " ... seq d_k"],
                 token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:

        '''
        x:(batch,seq,dim)
        '''
        *batch , s , _ =x.shape
        Q=self.q_proj(x)  
        K=self.k_proj(x)
        V=self.v_proj(x)
        q, k, v = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.n_head)
            for X in (Q, K, V)
        )  
           
        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq",
                                              torch.arange(s, device=x.device),
                                              b=[1] * len(batch))

        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        q = self.positional_encoder(q, token_positions)
        k = self.positional_encoder(k, token_positions)
    

        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,    # 如果用了 is_causal，这里传 None
            dropout_p=0.0, 
            is_causal=True     # 核心：自动启用 FlashAttention 的因果优化
            )
        attn_output = rearrange(attn_output, "... heads seq d_v -> ... seq (heads d_v)").contiguous()
    
        result = self.out_proj(attn_output)
        ''''
        scores : b h q k 
        v :  b h v d 

        '''
  
        return result

'''
FFN_SwiGLU(x) = (SiLU(xW1) * xW3)W2

'''
def silu_fn(d_model):
    return d_model * torch.sigmoid(d_model)


class SwiGLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int | None = None):
        super().__init__()
    
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            d_ff = 256 * ((d_ff + 256 - 1) // 256)

        self.d_ff = d_ff
        # 合并 W1 和 W3：输出维度变为 2 * d_ff
        self.w13 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False,)

    def forward(self, x: Tensor) -> Tensor:
        # 1. 一次性投影得到合并的特征
        # x_merged 形状: (batch, seq, 2 * d_ff)
        x_merged = self.w13(x)

        w1, w3 = rearrange(x_merged, "... (n d_ff) -> n ... d_ff", n=2)

        intermediate = silu_fn(w1) * w3
        # 4. 投影回原维度
        return self.w2(intermediate)

# class LayerNorm(nn.Module):
#     def __init__(self,d_model , eps=1e-5 ):
#         super().__init__()
#         self.d_model=d_model
#         self.eps=eps

#         self.weight=nn.Parameter(
#             torch.ones(d_model)
#         )

#         self.bias= nn.Parameter(
#             torch.zeros(d_model)
#         )
#     def forward(self,
#                 x:torch.Tensor
#             ):
#         in_dtype=x.dtype
#         x_float32=x.to(torch.float32)
#         mean=x_float32.mean(
#             dim=-1,
#             keepdim=-1
#         )
#         var=x_float32.var(
#             dim=-1,keepdim=-1,unbiased=False
#         )
#         x_normed=(x_float32-mean)*torch.rsqrt(var+self.eps)
#         result=x_normed * self.weight +self.bias

#         return result.to(in_dtype)
    

    
class RmsNorm(nn.Module):
    def __init__(self,d_model , eps=1e-5 ):
        super().__init__()
        self.d_model=d_model
        self.eps=eps

       
        self.weight=nn.Parameter(
            torch.ones(d_model)
        )

    def forward(self,x:torch.Tensor):
        in_dtype=x.dtype
        x_float32=x.to(torch.float32)
         
        rms=torch.rsqrt(
           ( x_float32**2).mean(dim=-1,keepdim=True)+self.eps
           )
        result=x_float32* rms * self.weight.to(torch.float32)

        return result.to(in_dtype)

class TransformerBlock(nn.Module):
    ''''
    两个 RMSNorm 层：一个用于 Attention 之前，一个用于 Feed-Forward (SwiGLU) 之前。
    一个 Causal Self-Attention 层：包含 RoPE 位置编码和FlashAttention。
    一个 Feed-Forward 层：SwiGLU。
    残差连接 (Residual Connections)
    '''         
    def __init__(self,
                 d_model:int,
                 n_head:int,
                 d_ff:int,
                 positional_encoder:RoPositionalEmbedding,
                 ):
        super().__init__()
        self.norm1= RmsNorm(d_model)
        self.norm2= RmsNorm(d_model)
        self.ffn  = SwiGLU (d_model,d_ff)
        self.attn = Causalmhattention(
            d_model,n_head,positional_encoder
        )
    def forward(self,
                x:torch.Tensor
                ):
        x_attn=self.attn(self.norm1(x))
        attn_out=x+x_attn
        x_ffn=self.ffn(self.norm2(attn_out))
        ffn_out=attn_out+x_ffn
        return ffn_out
    



class TransformerLM(nn.Module):

    def __init__(self,
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                n_head: int,
                d_ff: int,
                rope_theta: float,
                device=None
                ):
        
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        }
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_k = d_model // n_head

        self.positional_encoder = RoPositionalEmbedding(
            d_k=d_k,
            context_len=context_length,  
            theta=rope_theta,
            device=device
        )
        #2 堆叠transformer层

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_head=n_head,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder,
                    
                )
                for _ in range(num_layers)
            ]
        )
        # 3 输出层
        self.norm_final =  RmsNorm(d_model)
        self.out_head   =  Linear(d_model,vocab_size)
        

        #logger.info(f"number of non-embedding parameters: {get_num_params() / 1e6:.2f}M")

    def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        
        # batch,seq_len=x.shape
        x=self.token_embeddings(x)

       # x: [batch, seq_len] -> [batch,seq,d_model]s
        for layer in self.layers:
            x=layer(x)
        x=self.norm_final(x)

        return self.out_head(x)
    


    @torch.no_grad()

    def generate(
        self,
        x :torch.Tensor,
        max_new_tokens:int,
        eos_token_id:int |None=None,
        temperature:float=1.0,
        top_p:float | None = None
    ):
        if x.dim() ==1:
            x = x.unsqueeze(0)

        self.eval()
        original_sequence_length = x.size(-1)

        for _ in range(max_new_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x

            logits = self.forward(x)
            #输出 logits: 形状是 (Batch, original_sequence_length, vocab_size)。
            logits=logits[:,-1,:]


            logits = logits/(temperature+1e-8)
            
            if top_p is not None and top_p < 1.0:
                logits=self._top_p_filter(logits,top_p) 
            probs=torch.softmax(logits,dim=-1)

            next_token = torch.multinomial(probs,num_samples=1)

            x=torch.cat(
                (x,next_token),dim=1
            )

            if eos_token_id is not None and (next_token==eos_token_id) .all():
                break
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids
    
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """内部工具函数：执行 Top-P 截断"""
        # 对词表分值进行降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # 计算累积概率分布
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 创建掩码：我们要去掉累积概率超过 p 的 Token
        # 逻辑：保留最小的集合 V(p)，使其概率之和 >= p
        # 我们把所有超过 p 的位置标记为 True（需要移除）
        sorted_indices_to_remove = cumulative_probs > p
        
        # 做法是把标记位向右移动一格。
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 将被移除的 Token 分数设为负无穷
        # 这里需要利用 scatter 将排序后的掩码映射回原始词表索引位置
        indices_to_remove = sorted_indices_to_remove.scatter(1,
                                                            sorted_indices,
                                                            sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model   