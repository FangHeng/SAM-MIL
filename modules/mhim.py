import torch
import math
import numpy as np
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from modules.baseline import *
from functools import reduce
from operator import mul

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def get_pseudo_score_trans(classifier,feat,attention,to_out):
    b,h,n,d = feat.size()

    features = torch.einsum('hns,hn -> hns', feat.squeeze(0), attention.squeeze(0))
    features = rearrange(features, 'h n d -> n (h d)', h = h)
    features = to_out(features)

    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('gf,cf->cg', features, tweight)
    # 加上bias
    cam_maps += list(classifier.parameters())[-1].data[0]
    #cam_maps = classifier(feat.squeeze(0)).transpose(0,1)
    cam_maps = torch.nn.functional.softmax(cam_maps,dim=0)
    cam_maps,_ = torch.max(cam_maps.transpose(0,1),-1)
    return cam_maps.unsqueeze(0)

# 这里的本质是用img_cls 去给经过attention后的图块做分类，并不是所谓的cam
def get_pseudo_score(classifier, feat,attention):
    attention = attention.squeeze(0)
    features = torch.einsum('ns,n->ns', feat.squeeze(0), attention.squeeze(0))  ### n x fs
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('gf,cf->cg', features, tweight)
    # 加上bias
    cam_maps += list(classifier.parameters())[-1].data[0]
    #cam_maps = classifier(feat.squeeze(0)).transpose(0,1)
    cam_maps = torch.nn.functional.softmax(cam_maps,dim=0)
    cam_maps,_ = torch.max(cam_maps.transpose(0,1),-1)
    return cam_maps.unsqueeze(0)

def select_mask_fn(ps,attn,largest,mask_ratio,mask_ids_other=None,len_keep_other=None,cls_attn_topk_idx_other=None,random_ratio=1.,select_inv=False,msa_fusion='vote'):
        ps_tmp = ps
        mask_ratio_ori = mask_ratio
        mask_ratio = mask_ratio / random_ratio
        if mask_ratio > 1:
            random_ratio = mask_ratio_ori
            mask_ratio = 1.
            
        if mask_ids_other is not None:
            if cls_attn_topk_idx_other is None:
                cls_attn_topk_idx_other = mask_ids_other[:,len_keep_other:].squeeze()
                ps_tmp = ps - cls_attn_topk_idx_other.size(0)
        if len(attn.size()) > 2:
            if msa_fusion == 'mean':
                _,cls_attn_topk_idx = torch.topk(attn,int(np.ceil((ps_tmp*mask_ratio)) // attn.size(1)),largest=largest)
                cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3,-1))
            elif msa_fusion == 'vote':
                vote = attn.clone()
                vote[:] = 0
                
                _,idx = torch.topk(attn,k=int(np.ceil((ps_tmp*mask_ratio))),sorted=False,largest=largest)
                mask = vote.clone() 
                mask = mask.scatter_(2,idx,1) == 1
                vote[mask] = 1
                vote = vote.sum(dim=1)
                _,cls_attn_topk_idx = torch.topk(vote,k=int(np.ceil((ps_tmp*mask_ratio))),sorted=False)
                cls_attn_topk_idx = cls_attn_topk_idx[0]
        else:
            k = int(np.ceil((ps_tmp*mask_ratio)))
            _,cls_attn_topk_idx = torch.topk(attn,k,largest=largest)
            cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)
        
        # randomly 
        if random_ratio < 1.:
            random_idx = torch.randperm(cls_attn_topk_idx.size(0),device=cls_attn_topk_idx.device)
            cls_attn_topk_idx = torch.gather(cls_attn_topk_idx,dim=0,index=random_idx[:int(np.ceil((cls_attn_topk_idx.size(0)*random_ratio)))])
        
        # concat other masking idx
        if mask_ids_other is not None:
            cls_attn_topk_idx = torch.cat([cls_attn_topk_idx,cls_attn_topk_idx_other]).unique()

        # if cls_attn_topk_idx is not None:
        len_keep = ps - cls_attn_topk_idx.size(0)
        a = set(cls_attn_topk_idx.tolist())
        b = set(list(range(ps)))
        mask_ids =  torch.tensor(list(b.difference(a)),device=attn.device)
        if select_inv:
            mask_ids = torch.cat([cls_attn_topk_idx,mask_ids]).unsqueeze(0)
            len_keep = ps - len_keep
        else:
            mask_ids = torch.cat([mask_ids,cls_attn_topk_idx]).unsqueeze(0)

        return len_keep,mask_ids

def mask_fn(x, ids_shuffle=None,len_keep=None):
    """
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    assert ids_shuffle is not None

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    return x_keep

class MCA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, _q):
        kv = self.to_kv(x).chunk(2, dim = -1)
        q = self.to_q(_q)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Merge(nn.Module):
    def __init__(self, dim, heads = 8, merge_h_dim = 64, dropout = 0.1, k=10, g_q_mm=1., merge_ratio=0.2, global_q_enable=True,no_merge=False,g_q_grad=False,mask_type='random',**kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MCA(dim,heads,merge_h_dim,dropout)
        self.merge_k = k 
        self.no_merge = no_merge
        self.mask_type = mask_type
        if global_q_enable:
            if g_q_grad:
                self.global_q_grad = nn.Parameter(torch.zeros(1, k, dim),requires_grad=True)
                # copy from vpt@google
                val = math.sqrt(6. / float(3 * reduce(mul, (16,16), 1) + dim))  # noqa
                nn.init.uniform_(self.global_q_grad.data, -val, val)
            else:
                # self.global_q = nn.Parameter(torch.zeros(1, k, dim),requires_grad=False)
                self.global_q_grad = torch.zeros(1, k, dim)
            if g_q_mm != 1.:
                self.global_q_mm = nn.Parameter(torch.zeros(1, k, dim),requires_grad=False)
                # copy from vpt@google
                val = math.sqrt(6. / float(3 * reduce(mul, (16,16), 1) + dim))  # noqa
                nn.init.uniform_(self.global_q_mm.data, -val, val)
            else:
                self.global_q_mm = torch.zeros(1, k, dim)

            if g_q_grad and g_q_mm == 1.:
                self.global_q = self.global_q_grad
            elif not g_q_grad and g_q_mm != 1.:
                self.global_q = self.global_q_mm
        else:
            self.global_q = None

        self.g_q_mm = g_q_mm
        self.g_q_grad = g_q_grad
        self.merge_ratio = merge_ratio
        self.k = k

    def update_q_ema(self,new):
        self.global_q_mm.data.mul_(self.g_q_mm).add_(new,alpha=1. - self.g_q_mm)
    
    def merge(self,x):
        z = self.attn(self.norm(x),self.norm(self.global_q))
        if self.training and self.global_q is not None and self.g_q_mm != 1.:
            self.update_q_ema(z[:,:self.k])
        return z
    
    def masking(self,x,attn):
        B,L,C = x.shape
        merge_ratio = self.merge_ratio

        if self.mask_type == 'random':
            # random mask
            len_keep = int(L * merge_ratio)
            noise = torch.rand(L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
        elif self.mask_type == 'low':
            # low mask 
            len_keep,ids_shuffle = select_mask_fn(L,attn,False,1-merge_ratio)
            ids_shuffle = ids_shuffle.squeeze(0)
            
        ids_keep = ids_shuffle[:len_keep]
        ids_random = ids_shuffle[len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        x_masked = torch.gather(x, dim=1, index=ids_random.unsqueeze(-1).repeat(1, 1, C))

        return x_keep, x_masked

    def forward(self,x,attn=None):
        if self.training:
            x_keep, x_masked = self.masking(x,attn)
            if self.no_merge:
                if self.global_q is not None:
                    x_keep = torch.cat((x_keep,self.global_q),dim=1)
                return x_keep
            else:
                return torch.cat((x_keep,self.merge(x_masked)),dim=1)
        else:
            if not self.no_merge:
                return torch.cat((x,self.merge(x)),dim=1) 
            else:
                if self.global_q is not None:
                    x = torch.cat((x,self.global_q),dim=1)
                return x

class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self,temp_t=1.,temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool= True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t,dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss
        
class MHIM(nn.Module):
    def __init__(self, input_dim=1024,mlp_dim=512,mask_ratio=0,n_classes=2,temp_t=1.,temp_s=1.,dropout=0.25,act='relu',select_mask=True,select_inv=False,msa_fusion='vote',mask_ratio_h=0.,mrh_sche=None,mask_ratio_hr=0.,mask_ratio_l=0.,da_act='gelu',baseline='selfattn',head=8,attn_layer=0,attn2score=True,merge_enable=True,merge_k=1,merge_mm=0.9998,merge_ratio=0.1,merge_test=False,merge_mask_type='random'):
        super(MHIM, self).__init__()
 
        self.mask_ratio = mask_ratio
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_hr = mask_ratio_hr
        self.mask_ratio_l = mask_ratio_l
        self.select_mask = select_mask
        self.select_inv = select_inv
        self.msa_fusion = msa_fusion
        self.mrh_sche = mrh_sche
        self.attn_layer = attn_layer
        self.baseline = baseline
        self.merge_test=merge_test
        self.attn2score = attn2score
        self.head = head

        self.patch_to_emb = [nn.Linear(input_dim, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.merge = Merge(mlp_dim,k=merge_k,g_q_mm=merge_mm,merge_ratio=merge_ratio,mask_type=merge_mask_type) if merge_enable else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        if baseline == 'selfattn':
            self.online_encoder = SAttention(mlp_dim=mlp_dim,head=head)
        elif baseline == 'attn':
            self.online_encoder = DAttention(mlp_dim,da_act)
        elif baseline == 'dsmil':
            self.online_encoder = DSMIL(n_classes=n_classes,mlp_dim=mlp_dim,mask_ratio=mask_ratio,cls_attn=self.attn2score)

        self.predictor = nn.Linear(mlp_dim,n_classes)

        self.temp_t = temp_t
        self.temp_s = temp_s

        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t,self.temp_s)

        self.predictor_cl = nn.Identity()
        self.target_predictor = nn.Identity()

        self.apply(initialize_weights)

    def get_mask(self,ps,i,attn,mrh=None):
        
        # random mask, only for mhim_iccv
        if attn is not None and self.mask_ratio > 0.:
            len_keep,mask_ids = select_mask_fn(ps,attn,False,self.mask_ratio,select_inv=self.select_inv,random_ratio=0.001,msa_fusion=self.msa_fusion)
        else:
            len_keep,mask_ids = ps,None

        # low attention mask, only for mhim_iccv
        if attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep,mask_ids = select_mask_fn(ps,attn,False,self.mask_ratio_l,select_inv=self.select_inv,msa_fusion=self.msa_fusion)
            else:
                cls_attn_topk_idx_other = mask_ids[:,:len_keep].squeeze() if self.select_inv else mask_ids[:,len_keep:].squeeze()
                len_keep,mask_ids = select_mask_fn(ps,attn,False,self.mask_ratio_l,select_inv=self.select_inv,mask_ids_other=mask_ids,len_keep_other=ps,cls_attn_topk_idx_other = cls_attn_topk_idx_other,msa_fusion=self.msa_fusion)
        
        # high attention mask
        mask_ratio_h = self.mask_ratio_h
        if self.mrh_sche is not None:
            mask_ratio_h = self.mrh_sche[i]
        if mrh is not None:
            mask_ratio_h = mrh
        if mask_ratio_h > 0. :
            # mask high conf patch
            if mask_ids is None:
                len_keep,mask_ids = select_mask_fn(ps,attn,largest=True,mask_ratio=mask_ratio_h,len_keep_other=ps,random_ratio=self.mask_ratio_hr,select_inv=self.select_inv,msa_fusion=self.msa_fusion)
            else:
                cls_attn_topk_idx_other = mask_ids[:,:len_keep].squeeze() if self.select_inv else mask_ids[:,len_keep:].squeeze()
                
                len_keep,mask_ids = select_mask_fn(ps,attn,largest=True,mask_ratio=mask_ratio_h,mask_ids_other=mask_ids,len_keep_other=ps,cls_attn_topk_idx_other = cls_attn_topk_idx_other,random_ratio=self.mask_ratio_hr,select_inv=self.select_inv,msa_fusion=self.msa_fusion)

        return len_keep,mask_ids

    @torch.no_grad()
    def forward_teacher(self,x):

        x = self.patch_to_emb(x)
        x = self.dp(x)
        # teacher的merge应该是测试阶段的merge, only for ablation
        if self.merge_test:
            p = x.size(1)
            self.training = False
            x = self.merge(x)
            self.training = True

        if self.baseline == 'dsmil':
            _,x,attn = self.online_encoder(x,return_attn=True)
            if self.merge_test:
                attn = attn[:,:p]
        else:
            x,attn,act = self.online_encoder(x,return_attn=True,return_act=True)
            if self.merge_test:
                # transmil    n_layers,1,n_heads,L
                if type(attn) in (list,tuple):
                    attn = [ attn[i][:,:,:p] for i in range(len(attn)) ]
                else:
                    attn = attn[:,:p]

            if self.attn2score:
                if self.baseline == 'selfattn':
                    attn = get_pseudo_score_trans(self.predictor,act,attn[0],self.online_encoder.layer2.attn.to_out)
                else:
                    attn = get_pseudo_score(self.predictor,act,attn)
            else:
                if attn is not None and isinstance(attn,(list,tuple)):
                    attn = attn[self.attn_layer]
        ## softmax必须指定维数，不然会很奇怪的维度BUG，每次执行的维度随机
        return x,attn
    
    @torch.no_grad()
    def forward_test(self,x,return_attn=False,no_norm=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        if self.merge_test:
            x = self.merge(x)

        if return_attn:
            x,a = self.online_encoder(x,return_attn=True,no_norm=no_norm)
        else:
            x = self.online_encoder(x)

        if self.baseline == 'dsmil':
            pass
        else:    
            x = self.predictor(x)

        if return_attn:
            return x,a
        else:
            return x

    def pure(self,x):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if self.baseline == 'dsmil':
            x,_ = self.online_encoder(x)
        else:
            x = self.online_encoder(x)
            x = self.predictor(x)

        if self.training:
            return x, 0, ps,ps
        else:
            return x

    def forward_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat,teacher_cls_feat.detach())
        else:
            cls_loss = 0.
        
        return cls_loss

    def forward(self, x,attn=None,teacher_cls_feat=None,i=None):
        x = self.patch_to_emb(x)
        x = self.dp(x)

        ps = x.size(1)

        # get mask
        if self.select_mask:
            # mask high
            len_keep,mask_ids = self.get_mask(ps,i,attn)
            x = mask_fn(x,mask_ids,len_keep)
            ids_keep = mask_ids[:, :len_keep]
            if len(attn.size()) > 2:
                attn = torch.gather(attn, dim=-1, index=ids_keep.unsqueeze(1).repeat(1, self.head, 1))
            else:
                attn = torch.gather(attn, dim=-1, index=ids_keep)
        else:
            len_keep,mask_ids = ps,None

        x = self.merge(x,attn)

        len_keep = x.size(1)

        if self.baseline == 'dsmil':
            # forward online network
            student_logit,student_cls_feat= self.online_encoder(x)

            # cl loss
            cls_loss= self.forward_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)

            return student_logit, cls_loss,ps,len_keep
        
        else:
            # forward online network
            student_cls_feat = self.online_encoder(x)

            # prediction
            student_logit = self.predictor(student_cls_feat)

            # cl loss
            cls_loss= self.forward_loss(student_cls_feat=student_cls_feat,teacher_cls_feat=teacher_cls_feat)

            return student_logit, cls_loss,ps,len_keep
