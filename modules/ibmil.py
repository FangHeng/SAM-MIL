import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum
from math import ceil, sqrt
from einops import rearrange, reduce
import sys
# from .swin import SwinEncoder
sys.path.append("..")
from utils import group_shuffle



def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # ref from meituan
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class SwinEncoder(nn.Module):
    '''
    learnable mask token
    '''
    def __init__(self,mlp_dim=512,pos_pos=0,pos='none',peg_k=7,attn='ntrans',swin_window_num=8,drop_out=0.1,n_layers=2,n_heads=8,multi_scale=False,drop_path=0,pool='attn',da_act='tanh',reduce_ratio=0,ffn=False,ffn_act='gelu',mlp_ratio=4.,da_gated=False,da_bias=False,da_dropout=False,trans_dim=64,n_cycle=1,trans_conv=True,rpe=False,window_size=0,min_win_num=0,min_win_ratio=0,qkv_bias=True,shift_size=False,peg_bias=True,peg_1d=False,init=False,l2_n_heads=8,moe_fl_enable=False,**kwargs):
        super(SwinEncoder, self).__init__()
        
        # 不需要降维
        if reduce_ratio == 0:
            pass
        # 根据多尺度自动降维，维度前后不变
        elif reduce_ratio == -1:
            reduce_ratio = n_layers-1
        # 指定降维维度 (2的n次)
        else:
            pass

        self.final_dim = mlp_dim // (2**reduce_ratio) if reduce_ratio > 0 else mlp_dim
        if multi_scale:
            self.final_dim = self.final_dim * (2**(n_layers-1))

        self.pool = pool
        if pool == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        elif pool == 'attn':
            self.pool_fn = DAttention(self.final_dim,da_act,gated=da_gated,bias=da_bias,dropout=da_dropout)

        self.norm = nn.LayerNorm(self.final_dim)

        # l2_shortcut = kwargs.pop('l2_out_shortcut')
        # self.all_shortcut = kwargs.pop('all_shortcut')
        # l2_sc_ratio = kwargs.pop('l2_sc_ratio')
        # l2_conv = kwargs.pop('l2_conv_k')
        l2_shortcut = True
        self.all_shortcut = kwargs.pop('all_shortcut')
        l2_sc_ratio = 0
        l2_conv = 0
        if l2_conv == 0:
            l2_conv_k=0
            l2_conv=False
        else:
            l2_conv_k=l2_conv
            l2_conv=True
            
        # if attn == 'trans':
        self.layer1 = TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,need_down=multi_scale,need_reduce=reduce_ratio!=0,down_ratio=2**reduce_ratio,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_window=swin_window_num,trans_conv=trans_conv,rpe=rpe,window_size=window_size,min_win_num=min_win_num,min_win_ratio=min_win_ratio,qkv_bias=qkv_bias,shift_size=shift_size,moe_enable=moe_fl_enable,fl=True,shortcut=True,**kwargs)
        #self.layer2 = TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path)
        if reduce_ratio > 0:
            mlp_dim = mlp_dim // (2**reduce_ratio)

        if multi_scale:
            mlp_dim = mlp_dim*2

        if n_layers >= 2:
            self.layers = []
            for i in range(n_layers-2):
                self.layers += [TransLayer(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,need_down=multi_scale,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_window=swin_window_num,trans_conv=trans_conv,rpe=rpe,window_size=window_size,min_win_num=min_win_num,min_win_ratio=min_win_ratio,qkv_bias=qkv_bias) ]
                if multi_scale:
                    mlp_dim = mlp_dim*2
            # self.layers += [TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_window=swin_window_num,trans_conv=trans_conv,rpe=rpe,window_size=window_size,min_win_num=min_win_num,min_win_ratio=min_win_ratio,qkv_bias=qkv_bias,shift_size=shift_size,**kwargs)]
            kwargs.pop('conv_k')
            self.layers += [TransLayer(dim=mlp_dim,head=l2_n_heads,drop_out=drop_out,drop_path=drop_path,need_down=multi_scale,need_reduce=reduce_ratio!=0,down_ratio=2**reduce_ratio,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_window=swin_window_num,rpe=rpe,window_size=window_size,min_win_num=min_win_num,min_win_ratio=min_win_ratio,qkv_bias=qkv_bias,shift_size=shift_size,moe_enable=True,trans_conv=l2_conv,conv_k=l2_conv_k,shortcut=l2_shortcut,sc_ratio=l2_sc_ratio,**kwargs)]
            self.layers = nn.Sequential(*self.layers)
        else:
            self.layers = nn.Identity()

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim,k=peg_k,bias=peg_bias,conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if init:
            self.apply(initialize_weights)

class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        # 避免最后补出来的特征图小于卷积核，这里只能用zero padding
        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class PEG(nn.Module):
    def __init__(self, dim=512,k=7,bias=True,conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length >0:
            x = x[:,:-add_length]

        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class SINCOS(nn.Module):
    def __init__(self,embed_dim=512):
        super(SINCOS, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)
    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self,embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def forward(self, x):
        #B, N, C = x.shape
        B,H,W,C = x.shape
        # # padding
        # H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        # add_length = H * W - N
        # x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        # pos_embed = torch.zeros(1, H * W + 1, self.embed_dim)
        # pos_embed = self.get_2d_sincos_pos_embed(pos_embed.shape[-1], int(H), cls_token=True)
        #pos_embed = torch.from_numpy(self.pos_embed).float().unsqueeze(0).to(x.device)

        pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)

        # print(pos_embed.size())
        # print(x.size())
        x = x + pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        

        #x = x + pos_embed[:, 1:, :]

        # if add_length >0:
        #     x = x[:,:-add_length]

        return x

class APE(nn.Module):
    def __init__(self,embed_dim=512,num_patches=64):
        super(APE, self).__init__()
        self.absolute_pos_embed = nn.Parameter(torch.zeros( num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)
    
    def forward(self, x):
        B,H,W,C = x.shape
        return x + self.absolute_pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn1 = attn1[:,:,-n].unsqueeze(-2) @ attn2
            attn1 = (attn1 @ attn3)
        
            return out, attn1[:,:,0,-n+1:], v[:,:,-n+1:]

        return out
    
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class ProbAttentionLayer(nn.Module):
    def __init__(self,  d_model, n_heads, 
                 dim=None, mix=False,dropout=0.1):
        super(ProbAttentionLayer, self).__init__()

        dim = dim or (d_model//n_heads)
        dim = dim or (d_model//n_heads)

        self.inner_attention = ProbAttention(False,attention_dropout=dropout)

        self.to_qkv = nn.Linear(d_model, dim * n_heads * 3, bias = False)
        self.out_projection = nn.Linear(dim * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, x, return_attn=False):
        b, n, _, h = *x.shape, self.n_heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        out, attn = self.inner_attention(
            q,
            k,
            v,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(b, n, -1)

        if return_attn:
            return self.out_projection(out), attn
        else:
            return self.out_projection(out)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        B, L, C = x.shape
        # padding
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        _n = -H % 2
        H, W = H+_n, W+_n
        add_length = H * W - L
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)

        # H,W = int(L**0.5),int(L**0.5)
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8,drop_out=0.1,drop_path=0.,need_down=False,need_reduce=False,down_ratio=2,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,n_cycle=1,attn='ntrans',n_window=8,trans_conv=False,shift_size=False,window_size=0,rpe=False,min_win_num=0,min_win_ratio=0,qkv_bias=True,shortcut=True,sc_ratio='1',**kwargs):
        super().__init__()

        if need_reduce:
            self.reduction = nn.Linear(dim, dim//down_ratio, bias=False)
            dim = dim // down_ratio
        else:
            self.reduction = nn.Identity()
        
        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        trans_dim = trans_dim if trans_dim else dim // head
        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = trans_dim,  # dim // 8
                heads = head,
                num_landmarks = 256,    # number of landmarks dim // 2
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out
            )
        elif attn == 'ptrans':
            self.attn = ProbAttentionLayer(
                d_model=dim,
                dim=trans_dim,
                n_heads=head,
                dropout=drop_out,
            )
        elif attn == 'native':
            self.attn = Attention(
                dim=dim,
                heads=head,
                dim_head=trans_dim,
                dropout=drop_out,
                conv=trans_conv,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_out) if ffn else nn.Identity()

        self.downsample = PatchMerging(None,dim) if need_down else nn.Identity()

        self.n_cycle = n_cycle
        self.shortcut = shortcut
        self.sc_ratio = sc_ratio
        self.sl_sc_disable_test = True
        if sc_ratio == 'param':
            self._sc_ratio = nn.Parameter(
                torch.zeros(1,)
            )
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self._sc_ratio = float(sc_ratio)

    def forward(self,x,need_attn=False):
        attn = None
        for i in range(self.n_cycle):
            x,attn = self.forward_trans(x,need_attn=need_attn)
        
        if need_attn:
            return x,attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None

        x = self.reduction(x)
        B, L, C = x.shape
        
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        # if self.shortcut:
        #     if self.sc_ratio == 'param':
        #         x = self.sigmoid(self._sc_ratio)*x+self.drop_path(z)
            # else:
            #     x = self._sc_ratio*x+self.drop_path(z)
        # else:
        #     x = self.drop_path(z)
        # if self.sl_sc_disable_test and not self.training:
        #     x = self.drop_path(z)
        # else:
        x = x+self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.downsample(x)

        return x,attn

class FCLayer(nn.Module):
    def __init__(self, dropout=0.25,act='relu',in_size=1024,swin=False,swin_convk=15,swin_moek=3,swin_as=False,swin_md=False):
        super(FCLayer, self).__init__()
        self.embed = [nn.Linear(in_size, 512)]
        # self.embed.append(SwinEncoder(attn='swin',pool='none'))
        # self.embed = nn.ModuleList([nn.Linear(1024, 512)])
        
        if act.lower() == 'gelu':
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(dropout)]
        self.swin = SwinEncoder(attn='swin',pool='none',n_layers=2,conv_k=swin_convk,moe_k=swin_moek,all_shortcut=swin_as,moe_mask_diag=swin_md,init=True,drop_path=0) if swin else nn.Identity()
        self.embed = nn.Sequential(*self.embed)

    def forward(self, feats):
        feats = self.embed(feats)

        return self.swin(feats)

class Attention(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D,bias=bias)]

        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K,bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self,x,no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)
        
        if no_norm:
            return x,A_ori
        else:
            return x,A


class AttentionGated(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

    def forward(self, x,no_norm=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        if no_norm:
            return x,A_ori
        else:
            return x,A

class DAttention(nn.Module):
    def __init__(self,input_dim=512,act='relu',mask_ratio=0.,gated=False,bias=False,dropout=False,**kwargs):
        super(DAttention, self).__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim,act,bias,dropout)
        else:
            self.attention = Attention(input_dim,act,bias,dropout)

        self.mask_ratio = mask_ratio

    def random_masking(self, x, mask_ratio,ids_shuffle=None,len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        N, L, D = x.shape  # batch, length, dim
        if ids_shuffle is None:
            # sort noise for each sample
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        else:
            _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,no_norm=False,mask_enable=False,classifier=None,**kwags):
        #b,p,n = x.size()

        if mask_enable and (self.mask_ratio > 0. or mask_ids is not None):
            x, _,_ = self.random_masking(x,self.mask_ratio,mask_ids,len_keep)

        # x_shortcut = x.clone()

        act = x.clone()
        x,attn = self.attention(x,no_norm)

        # if classifier is not None:
        #     attn = get_cam_1d(classifier,x_shortcut,attn)

        # if return_attn:
        #     return x.squeeze(1),attn.squeeze(1)
        # else:   
        #     return x.squeeze(1)

        return x.squeeze(1),attn.squeeze(1),act.squeeze(1)

class Dattention_ori(nn.Module):
    def __init__(self,out_dim=2,in_size=1024,dropout=0.25,confounder_path=False,**kwargs):
        super(Dattention_ori,self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.embedding = FCLayer(in_size=in_size,dropout=dropout,**kwargs)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.head = nn.Linear(512,out_dim)

        if confounder_path: 
            print('deconfounding')
            self.confounder_path = confounder_path
            conf_list = []
            if isinstance(confounder_path,list):
                for i in confounder_path:
                    conf_list.append(torch.from_numpy(np.load(i)).view(-1,512).float())
            else:
                conf_list.append(torch.from_numpy(np.load(confounder_path)).view(-1,512).float())
            conf_tensor = torch.cat(conf_list, 0) 
            conf_tensor_dim = conf_tensor.shape[-1]
            self.register_buffer("confounder_feat",conf_tensor)
            joint_space_dim = 128
            dropout_v = 0.5
            # self.confounder_W_q = nn.Linear(in_size, joint_space_dim)
            # self.confounder_W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.W_q = nn.Linear(512, joint_space_dim)
            self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.head =  nn.Linear(self.L*self.K+conf_tensor_dim, out_dim)
            self.dropout = nn.Dropout(dropout_v)

        self.apply(initialize_weights)
        
    def forward(self,x):
        x=x.squeeze()
        x = self.embedding(x) # 1024->512
        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        x = torch.mm(A, x)  # KxL

        if self.confounder_path:
            device = x.device
            # bag_q = self.confounder_W_q(M)
            # conf_k = self.confounder_W_k(self.confounder_feat)
            bag_q = self.W_q(x)
            conf_k = self.W_k(self.confounder_feat)
            deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
            deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            conf_feats = torch.mm(deconf_A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
            x = torch.cat((x,conf_feats),dim=1)

        x = self.head(x)
        return x

if __name__ == "__main__":
    x=torch.rand(5,3,64,64).cuda()

