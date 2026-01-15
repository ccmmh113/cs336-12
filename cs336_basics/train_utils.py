import torch
import math
from torch.optim  import Optimizer
from collections.abc import Iterable
import numpy as np

'''
交叉熵实现

'''
# def log_sum_exp(x, dim=-1):
#     # 1. 找到最大值 c，keepdim=True 
#     c_max, _ = torch.max(x, dim=dim, keepdim=True)
    
#     # 2. 减去最大值，计算指数，求和
#     sum_exp = torch.sum(torch.exp(x - c_max), dim=dim, keepdim=True)
    
#     # 3. 加上对数，最后加上之前提出来的最大值 c
#     lse = c_max + torch.log(sum_exp)
    
#     # 去除 keepdim 产生的多余维度
#     return lse.squeeze(dim)

# def cross_entropy(logits,
#                   targets):
#     '''
#     logits:(batch,seq,vocab_size)
#     target:(batch,seq)
#     '''
#     log=log_sum_exp(logits)
#     loss=log-targets

#     return torch.mean(loss) #转换成标量
'''
优化器 Adamw

'''

class Adamw(Optimizer):
    def __init__(self, params, lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        if lr <0.0:
            raise ValueError(f"Invalid learning rate: {lr}"
                            )
        if not 0.0 <= betas[0] < 1.0:
             raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
             raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps <0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        loss=None
        for group in self.param_groups:

            beta1,beta2 =group['betas']
            eps  =  group['eps']
            lr=group['lr']
            wd=group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad=p.grad
                state=self.state[p]
                if len(state)==0:
                    state['step']=0
                    state['exp_avg'] = torch.zeros_like(p,memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p,memory_format=torch.preserve_format)
                exp_avg,exp_avg_sq=state['exp_avg'],state['exp_avg_sq']
                state['step']+=1
                t=state['step']
                '''
                更新矩估计
                '''
                # mt​=β1​⋅mt−1​+(1−β1​)⋅gt​
                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                # vt=β2​⋅vt-1 +(1-β2)​⋅gt*gt
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                
                #消除初始值为 0 的偏置
                bias_correction1 = 1- beta1**t
                bias_correction2=1-beta2**t
                #步长
                step_size = lr * math.sqrt(bias_correction2)/bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)
                if wd!=0:
                    p.add_(p,alpha=-lr*wd)
        return loss
    


''''

余弦退火调整学习率

'''
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    

    if it < warmup_iters:
        # Warm-up 阶段：线性增加学习率
        lr = (it / warmup_iters) * max_learning_rate

    elif it <= cosine_cycle_iters:
        # Cosine Annealing 阶段：余弦函数衰减
        t = it - warmup_iters
        T = cosine_cycle_iters - warmup_iters
        cos_value = np.cos(np.pi * t / T)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cos_value)
    else:
        # Post-annealing 阶段：学习率保持最小值
        lr = min_learning_rate

    return lr

'''
梯度裁剪

parameters  :  模型的所有参数(model.parameters())
max_l2_norm :  允许的最大梯度L2范数(M)

'''
def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
  
    parameters_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(parameters_with_grad) == 0:
        return
    # 提取所有梯度的 L2 范数并拼接为一个 1D 向量，然后求和开根号，速度快很多
    total_norm = torch.norm(
                            torch.stack([torch.norm(p.grad.detach(), 2) 
                                         for p in parameters_with_grad]), 
                            2
                        )
    
    clip_coef = max_l2_norm / (total_norm + 1e-6)  
    
    if clip_coef < 1.0:
        for p in parameters_with_grad:
            p.grad.mul_(clip_coef)