import copy
import torch
import numpy as np
from torch import nn
from einops import repeat
import torchvision.models as models
from modules.emb_position import *
from modules.transformer import *
from modules.mlp import *
from modules.datten import *
from modules.swin_atten import *
import torch.nn.functional as F
from modules.translayer import *
from modules.datten import DAttention
import math


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # ref from meituan
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def initialize_weights_noic(module):
    for m in module.modules():
        # if isinstance(m, nn.Conv2d):
        #     # ref from huggingface
        #         m.weight.data.normal_(0, 0.02)
        #     #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinEncoder(nn.Module):
    '''
    learnable mask token
    '''

    def __init__(self, mlp_dim=512, pos_pos=0, pos='none', peg_k=7, attn='ntrans', swin_window_num=8, drop_out=0.1,
                 n_layers=2, n_heads=8, multi_scale=False, drop_path=0, pool='attn', da_act='tanh', reduce_ratio=0,
                 ffn=False, ffn_act='gelu', mlp_ratio=4., da_gated=False, da_bias=False, da_dropout=False, trans_dim=64,
                 n_cycle=1, trans_conv=True, rpe=False, window_size=0, min_win_num=0, min_win_ratio=0, qkv_bias=True,
                 shift_size=False, peg_bias=True, peg_1d=False, init=False, l2_n_heads=8, moe_fl_enable=False,
                 **kwargs):
        super(SwinEncoder, self).__init__()

        # 不需要降维
        if reduce_ratio == 0:
            pass
        # 根据多尺度自动降维，维度前后不变
        elif reduce_ratio == -1:
            reduce_ratio = n_layers - 1
        # 指定降维维度 (2的n次)
        else:
            pass

        self.final_dim = mlp_dim // (2 ** reduce_ratio) if reduce_ratio > 0 else mlp_dim
        if multi_scale:
            self.final_dim = self.final_dim * (2 ** (n_layers - 1))

        self.pool = pool
        if pool == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        elif pool == 'attn':
            self.pool_fn = DAttention(self.final_dim, da_act, gated=da_gated, bias=da_bias, dropout=da_dropout)

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
            l2_conv_k = 0
            l2_conv = False
        else:
            l2_conv_k = l2_conv
            l2_conv = True

        # if attn == 'trans':
        self.layer1 = TransLayer1(dim=mlp_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path,
                                  need_down=multi_scale, need_reduce=reduce_ratio != 0, down_ratio=2 ** reduce_ratio,
                                  ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio, trans_dim=trans_dim, n_cycle=n_cycle,
                                  attn=attn, n_window=swin_window_num, trans_conv=trans_conv, rpe=rpe,
                                  window_size=window_size, min_win_num=min_win_num, min_win_ratio=min_win_ratio,
                                  qkv_bias=qkv_bias, shift_size=shift_size, moe_enable=moe_fl_enable, fl=True,
                                  shortcut=True, **kwargs)
        # self.layer2 = TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path)
        if reduce_ratio > 0:
            mlp_dim = mlp_dim // (2 ** reduce_ratio)

        if multi_scale:
            mlp_dim = mlp_dim * 2

        if n_layers >= 2:
            self.layers = []
            for i in range(n_layers - 2):
                self.layers += [TransLayer1(dim=mlp_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path,
                                            need_down=multi_scale, ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio,
                                            trans_dim=trans_dim, n_cycle=n_cycle, attn=attn, n_window=swin_window_num,
                                            trans_conv=trans_conv, rpe=rpe, window_size=window_size,
                                            min_win_num=min_win_num, min_win_ratio=min_win_ratio, qkv_bias=qkv_bias)]
                if multi_scale:
                    mlp_dim = mlp_dim * 2
            # self.layers += [TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_window=swin_window_num,trans_conv=trans_conv,rpe=rpe,window_size=window_size,min_win_num=min_win_num,min_win_ratio=min_win_ratio,qkv_bias=qkv_bias,shift_size=shift_size,**kwargs)]
            kwargs.pop('conv_k')
            self.layers += [
                TransLayer1(dim=mlp_dim, head=l2_n_heads, drop_out=drop_out, drop_path=drop_path, need_down=multi_scale,
                            need_reduce=reduce_ratio != 0, down_ratio=2 ** reduce_ratio, ffn=ffn, ffn_act=ffn_act,
                            mlp_ratio=mlp_ratio, trans_dim=trans_dim, n_cycle=n_cycle, attn=attn,
                            n_window=swin_window_num, rpe=rpe, window_size=window_size, min_win_num=min_win_num,
                            min_win_ratio=min_win_ratio, qkv_bias=qkv_bias, shift_size=shift_size, moe_enable=True,
                            trans_conv=l2_conv, conv_k=l2_conv_k, shortcut=l2_shortcut, sc_ratio=l2_sc_ratio, **kwargs)]
            self.layers = nn.Sequential(*self.layers)
        else:
            self.layers = nn.Identity()

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if init:
            self.apply(initialize_weights)

    def forward(self, x, no_pool=False, return_trans_attn=False, return_attn=False, no_norm=False):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            shape_len = 4
        batch, num_patches, C = x.shape  # 直接是特征
        x_shortcut = x
        patch_idx = 0
        if self.pos_pos == -2:
            x = self.pos_embedding(x)

        # cls_token
        if self.pool == 'cls_token':
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
            x = torch.cat((cls_tokens, x), dim=1)
            patch_idx = 1

        if self.pos_pos == -1:
            x[:, patch_idx:, :] = self.pos_embedding(x[:, patch_idx:, :])

        # translayer1
        if return_trans_attn:
            x, trans_attn = self.layer1(x, True)
        else:
            x = self.layer1(x)
            trans_attn = None
        # print(x)
        # 加位置编码
        if self.pos_pos == 0:
            x[:, patch_idx:, :] = self.pos_embedding(x[:, patch_idx:, :])

        # translayer2
        for i, layer in enumerate(self.layers.children()):
            if return_trans_attn:
                x, trans_attn = layer(x, return_trans_attn)
            else:
                x = layer(x)
                trans_attn = None

        if self.all_shortcut:
            x = x + x_shortcut

        # ---->cls_token
        x = self.norm(x)

        if no_pool or self.pool == 'none':
            if shape_len == 2:
                x = x.squeeze(0)
            elif shape_len == 4:
                x = x.transpose(1, 2)
                x = x.reshape(batch, C, int(num_patches ** 0.5), int(num_patches ** 0.5))
            return x

        if self.pool == 'cls_token':
            logits = x[:, 0, :]
        elif self.pool == 'avg':
            logits = x.mean(dim=1)
        elif self.pool == 'attn':
            if return_attn:
                logits, a, _ = self.pool_fn(x, return_attn=True, no_norm=no_norm)
            else:
                logits, _, _ = self.pool_fn(x)

        else:
            logits = x

        if shape_len == 2:
            logits = logits.squeeze(0)
        elif shape_len == 4:
            logits = logits.transpose(1, 2)
            logits = logits.reshape(batch, C, int(num_patches ** 0.5), int(num_patches ** 0.5))

        if return_attn:
            return logits, a, trans_attn

        else:
            return logits, trans_attn


class Swin(nn.Module):
    def __init__(self, input_dim=1024, mlp_dim=512, act='relu', n_classes=2, dropout=0.25, pos_pos=0, n_robust=0,
                 pos='ppeg', peg_k=7, attn='trans', pool='attn', swin_window_num=8, n_layers=2, n_heads=8,
                 multi_scale=False, drop_path=0., da_act='relu', trans_dropout=0.1, reduce_ratio=0, ffn=False,
                 ffn_act='gelu', ic=False, mlp_ratio=4., da_gated=False, da_bias=False, da_dropout=False, trans_dim=64,
                 n_cycle=1, trans_conv=False, rpe=False, window_size=0, min_win_num=0, min_win_ratio=0, qkv_bias=True,
                 shift_size=False, **kwargs):
        super(Swin, self).__init__()

        self.patch_to_emb = [nn.Linear(input_dim, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.swin_window_num = swin_window_num

        self.online_encoder = SwinEncoder(mlp_dim=mlp_dim, pos_pos=pos_pos, pos=pos, peg_k=peg_k, attn=attn,
                                          swin_window_num=swin_window_num, n_layers=n_layers, n_heads=n_heads,
                                          multi_scale=multi_scale, drop_path=drop_path, pool=pool, da_act=da_act,
                                          drop_out=trans_dropout, reduce_ratio=reduce_ratio, ffn=ffn, ffn_act=ffn_act,
                                          mlp_ratio=mlp_ratio, da_gated=da_gated, da_bias=da_bias,
                                          da_dropout=da_dropout, trans_dim=trans_dim, n_cycle=n_cycle,
                                          trans_conv=trans_conv, rpe=rpe, window_size=window_size,
                                          min_win_num=min_win_num, min_win_ratio=min_win_ratio, qkv_bias=qkv_bias,
                                          shift_size=shift_size, **kwargs)

        self.predictor = nn.Linear(self.online_encoder.final_dim, n_classes)

        if ic:
            self.apply(initialize_weights)
        else:
            self.apply(initialize_weights_noic)

        if n_robust > 0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            if ic:
                self.apply(initialize_weights)
            else:
                self.apply(initialize_weights_noic)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024, 512)) for i in range(n_robust)]

    def crop(self, x):
        B, L, C = x.shape
        H, W = int(np.floor(np.sqrt(L))), int(np.floor(np.sqrt(L)))
        _n = H % self.swin_window_num
        H, W = H - _n, W - _n
        crop_length = L - H * W
        if crop_length > 0:
            if self.training:
                _indices = list(range(L))
                np.random.shuffle(_indices)
                _indices = _indices[:crop_length]
            else:
                _indices = np.linspace(0, L - 1, crop_length, dtype=int)
            _mask = torch.ones((B, L, C), device=x.device)
            _mask.scatter_(1, torch.tensor(_indices, device=x.device).unsqueeze(0).unsqueeze(-1).repeat((B, 1, C)), 0)
            _mask = _mask == 1
            x = x[_mask].view(1, -1, C)
        return x

    def forward(self, x, return_attn=False, no_norm=False, return_trans_attn=False):
        x = self.patch_to_emb(x)  # n*512
        x = self.dp(x)

        ps = x.size(1)

        # forward online network
        if return_attn:
            x, a, t_a = self.online_encoder(x, return_attn=True, no_norm=no_norm, return_trans_attn=return_trans_attn)
        else:
            x, t_a = self.online_encoder(x, return_trans_attn=return_trans_attn)

        # prediction
        logits = self.predictor(x)

        if return_attn:
            if return_trans_attn:
                return logits, a, t_a
            else:
                return logits, a
        else:
            if return_trans_attn:
                return logits, t_a
            else:
                return logits
