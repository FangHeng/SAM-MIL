
import copy
import torch
import numpy as np
from torch import nn
from einops import repeat, rearrange
import torchvision.models as models
from modules.emb_position import *
from modules.transformer import *
from modules.mlp import *
from modules.datten import *
from modules.swin_atten import *
import torch.nn.functional as F
from modules.translayer import *
from .swin import SwinEncoder
from functools import reduce
from operator import mul

sys.path.append("..")
from utils import cosine_scheduler


def get_cam_1d_trans(classifier, feat, attention, label, to_out):
    b, h, n, d = feat.size()

    features = torch.einsum('hns,hn -> hns', feat.squeeze(0), attention.squeeze(0))
    features = rearrange(features, 'h n d -> n (h d)', h=h)
    features = to_out(features)

    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('gf,cf->cg', features, tweight)
    # bias
    cam_maps += list(classifier.parameters())[-1].data[0]
    # cam_maps = classifier(feat.squeeze(0)).transpose(0,1)
    cam_maps = torch.nn.functional.softmax(cam_maps, dim=0)
    if label is not None:
        cam_maps = cam_maps[label]
    else:
        cam_maps, _ = torch.max(cam_maps.transpose(0, 1), -1)
    return cam_maps.unsqueeze(0)


def get_cam_1d(classifier, feat, attention, label):
    attention = attention.squeeze(0)
    features = torch.einsum('ns,n->ns', feat.squeeze(0), attention.squeeze(0))  ### n x fs
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('gf,cf->cg', features, tweight)
    # bias
    cam_maps += list(classifier.parameters())[-1].data[0]
    # cam_maps = classifier(feat.squeeze(0)).transpose(0,1)
    cam_maps = torch.nn.functional.softmax(cam_maps, dim=0)
    if label is not None:
        cam_maps = cam_maps[label]
    else:
        cam_maps, _ = torch.max(cam_maps.transpose(0, 1), -1)
    return cam_maps.unsqueeze(0)


def initialize_weights(module):
    for m in module.modules():
        # ref from https://github.com/Meituan-AutoML/Twins/blob/4700293a2d0a91826ab357fc5b9bc1468ae0e987/gvt.py#L356
        # if isinstance(m, nn.Conv2d):
        #     fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     fan_out //= m.groups
        #     m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self, temp_t=1., temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool = True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t, dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss


class MCA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, _q):
        kv = self.to_kv(x).chunk(2, dim=-1)
        q = self.to_q(_q)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class sam_mil(nn.Module):
    def __init__(self, input_dim=1024, mlp_dim=512, mask_ratio=0, n_classes=2, temp_t=1., temp_s=1., dropout=0.25,
                 n_robust=0, act='relu', select_mask=False, select_inv=False, select_type='mean', mask_ratio_h=0.,
                 mrh_sche=None, mrh_type='tea', mask_ratio_hr=0., mask_ratio_l=0., da_act='gelu', backbone='selfattn',
                 da_gated=False, da_bias=False, da_dropout=False, dsmil_cls_attn=True, dsmil_flat_B=False,
                 dsmil_attn_index='max', rrt=False, rrt_k=15, trans_conv=True, attn2score=False, score_label=-1,
                 cl_type='feat', head=8, merge_enable=False, merge_k=10, merge_mm=0.9998, merge_ratio=0.1,
                 merge_g_q=False, test_merge=False, attn_layer=0, mask_non_group_feat=False, mask_by_seg_area=False,
                 sigmoid_k=0.0005, sigmoid_A0=5000, **kwargs):
        super(sam_mil, self).__init__()

        self.mask_ratio = mask_ratio
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_hr = mask_ratio_hr
        self.mask_ratio_l = mask_ratio_l
        self.select_mask = select_mask
        self.select_inv = select_inv
        self.select_type = select_type
        self.mrh_sche = mrh_sche
        self.mrh_type = mrh_type
        self.attn2score = attn2score
        self.score_label = score_label
        self.test_merge = test_merge
        self.attn_layer = attn_layer

        self.patch_to_emb = [nn.Linear(input_dim, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.rrt = SwinEncoder(attn='swin', pool='none', n_heads=8, swin_window_num=8, conv_k=rrt_k, init=True,
                               trans_conv=trans_conv) if rrt else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.backbone = backbone

        self.online_encoder = Backbone(mlp_dim, backbone, attn2score, head=head, act=da_act, gated=da_gated,
                                       bias=da_bias, dropout=da_dropout, flat_B=dsmil_flat_B,
                                       attn_index=dsmil_attn_index)

        self.predictor = nn.Linear(mlp_dim, n_classes)

        self.temp_t = temp_t
        self.temp_s = temp_s

        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t, self.temp_s)
        self.cl_type = cl_type

        self.predictor_cl = nn.Identity()
        self.target_predictor = nn.Identity()

        self.apply(initialize_weights)

        self.mask_non_group_feat = mask_non_group_feat
        self.mask_by_seg_area = mask_by_seg_area

        self.sigmoid_k = sigmoid_k
        self.sigmoid_A0 = sigmoid_A0

        if n_robust > 0:
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            self.apply(initialize_weights)
            torch.random.set_rng_state(rng)

            [torch.rand((1024, 512)) for i in range(n_robust)]

    def adjusted_sigmoid(self, area):
        """
        Adjusted sigmoid function to handle large area values with sensitivity.

        :param area: The area value to calculate the sigmoid for.
        :param k: The slope control parameter, adjusting the sensitivity of the function.
        :param A0: The center point or offset of the function, adjusting for the range of area values.
        :return: The calculated sigmoid value.
        """
        k = self.sigmoid_k
        A0 = self.sigmoid_A0
        return 1 / (1 + np.exp(-k * (area - A0)))

    def select_mask_fn(self, ps, attn, largest, mask_ratio, mask_ids_other=None, len_keep_other=None,
                       cls_attn_topk_idx_other=None, random_ratio=1., select_inv=False):
        ps_tmp = ps
        mask_ratio_ori = mask_ratio
        mask_ratio = mask_ratio / random_ratio
        if mask_ratio > 1:
            random_ratio = mask_ratio_ori
            mask_ratio = 1.

        if mask_ids_other is not None:
            if cls_attn_topk_idx_other is None:
                cls_attn_topk_idx_other = mask_ids_other[:, len_keep_other:].squeeze()
                ps_tmp = ps - cls_attn_topk_idx_other.size(0)
        if len(attn.size()) > 2:
            # mean
            if self.select_type == 'mean':
                _, cls_attn_topk_idx = torch.topk(attn, int(np.ceil((ps_tmp * mask_ratio)) // attn.size(1)),
                                                  largest=largest)
                cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3, -1))
            # vote
            elif self.select_type == 'vote':
                vote = attn.clone()
                vote[:] = 0

                _, idx = torch.topk(attn, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False, largest=largest)
                mask = vote.clone()
                mask = mask.scatter_(2, idx, 1) == 1
                vote[mask] = 1
                vote = vote.sum(dim=1)
                _, cls_attn_topk_idx = torch.topk(vote, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False)
                # print(cls_attn_topk_idx.size())
                cls_attn_topk_idx = cls_attn_topk_idx[0]
        else:
            k = int(np.ceil((ps_tmp * mask_ratio)))
            _, cls_attn_topk_idx = torch.topk(attn, k, largest=largest)
            cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)

        cls_attn_keep_idx = None
        # random mask
        if random_ratio < 1.:
            random_idx = torch.randperm(cls_attn_topk_idx.size(0), device=cls_attn_topk_idx.device)

            num_to_select = int(np.floor(cls_attn_topk_idx.size(0) * random_ratio))
            if num_to_select > 0:
                cls_attn_topk_idx = torch.gather(cls_attn_topk_idx, dim=0, index=random_idx[:num_to_select])
            else:
                cls_attn_topk_idx = torch.tensor([], dtype=cls_attn_topk_idx.dtype, device=cls_attn_topk_idx.device)

        # merge other part
        if mask_ids_other is not None:
            cls_attn_topk_idx = torch.cat([cls_attn_topk_idx, cls_attn_topk_idx_other]).unique()

        # if cls_attn_topk_idx is not None:
        len_keep = ps - cls_attn_topk_idx.size(0)
        a = set(cls_attn_topk_idx.tolist())
        b = set(list(range(ps)))

        # mask ids
        mask_ids = torch.tensor(list(b.difference(a)), device=attn.device)
        if select_inv:
            mask_ids = torch.cat([cls_attn_topk_idx, mask_ids]).unsqueeze(0)
            len_keep = ps - len_keep
        else:
            mask_ids = torch.cat([mask_ids, cls_attn_topk_idx]).unsqueeze(0)

        return len_keep, mask_ids

    def get_mask(self, ps, i, attn, mrh=None, pred_correct=False, is_group_feat=None, relative_area=None):
        if attn is not None and isinstance(attn, (list, tuple)):
            if self.attn_layer == -1:
                attn = attn[1]
            else:
                attn = attn[self.attn_layer]
        else:
            attn = attn

        # update ps according to is_group_feat
        if self.mask_non_group_feat and is_group_feat is not None:
            # is_group_feat_tensor = torch.from_numpy(is_group_feat).to(attn.device)
            non_group_indices = torch.where(is_group_feat == 0)[0]  # get the indices of non-group features

            # get the indices and length of group features
            group_indices = torch.where(is_group_feat == 1)[0]
            group_ps_len = group_indices.size(0)

            # only refactor to the feature
            # The first dimension of attn should not be changed, we only need to use the index on the second dimension
            attn = attn[:, non_group_indices]

            # update ps to non-group features length
            ps = non_group_indices.size(0)

        # random mask
        if attn is not None and self.mask_ratio > 0.:
            if self.mask_by_seg_area and relative_area is not None:
                # get unique areas
                unique_areas = torch.unique(relative_area)
                # initialize the final retained and masked indices
                final_retained_idxs = []
                final_masked_idxs = []
                # initialize the total length of retained indices
                len_keep = 0

                for area in unique_areas:
                    area_value = area.item()

                    # skip the area if it is 0
                    if area_value == 0:
                        continue

                    # get the indices of the current area
                    area_indices = torch.where(relative_area[0] == area)[0]

                    # fliter out the area with no indices
                    area_attn = attn[:, area_indices]
                    # calculate the number of patches in the current area
                    area_ps = area_indices.size(0)
                    # adjust the mask ratio according to the area value
                    adjusted_random_ratio = self.adjusted_sigmoid(area.item()) * self.mask_ratio

                    area_len_keep, area_mask_ids = self.select_mask_fn(area_ps, area_attn, False, adjusted_random_ratio,
                                                                       select_inv=self.select_inv,
                                                                       random_ratio=0.001)

                    # split the area indices into retained and masked indices
                    retained_idxs_local = area_mask_ids[0][:area_len_keep] # retained indices
                    masked_idxs_local = area_mask_ids[0][area_len_keep:]  # masked indices

                    # get the global indices of retained and masked indices
                    retained_idxs_global = area_indices[retained_idxs_local.squeeze().long()]
                    masked_idxs_global = area_indices[masked_idxs_local.squeeze().long()]

                    # check if retained_idxs_global is zero-dimensional and process accordingly
                    if retained_idxs_global.dim() == 0:
                        retained_idxs_global = retained_idxs_global.unsqueeze(0)
                    else:
                        retained_idxs_global = retained_idxs_global

                    # check if masked_idxs_global is zero-dimensional and process accordingly
                    # For an empty Tensor, dim() also returns 1 (since it is considered [0]), so it mainly deals with the non-empty zero-dimensional case
                    if masked_idxs_global.dim() == 0:
                        masked_idxs_global = masked_idxs_global.unsqueeze(0)
                    else:
                        masked_idxs_global = masked_idxs_global

                    # save the retained and masked indices
                    final_retained_idxs.append(retained_idxs_global.unsqueeze(0))
                    final_masked_idxs.append(masked_idxs_global.unsqueeze(0))

                    # update the total length of retained indices
                    len_keep += area_len_keep

                # merge the retained and masked indices
                retained_idxs = torch.cat(final_retained_idxs, dim=1)
                masked_idxs = torch.cat(final_masked_idxs, dim=1)


                # merge all the retained and masked indices
                mask_ids = torch.cat([retained_idxs, masked_idxs], dim=1)
            else:
                len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio, select_inv=self.select_inv,
                                                         random_ratio=0.001)
        else:
            len_keep, mask_ids = ps, None

        # low mask
        if attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:, :len_keep].squeeze() if self.select_inv else mask_ids[:,
                                                                                                   len_keep:].squeeze()
                len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv,
                                                         mask_ids_other=mask_ids, len_keep_other=ps,
                                                         cls_attn_topk_idx_other=cls_attn_topk_idx_other)

        # mask high attention
        mask_ratio_h = self.mask_ratio_h
        if self.mrh_sche is not None:
            mask_ratio_h = self.mrh_sche[i]
        if mrh is not None:
            mask_ratio_h = mrh
        if mask_ratio_h > 0. and not pred_correct and attn is not None:
            # mask high conf patch
            if mask_ids is None:
                len_keep, mask_ids = self.select_mask_fn(ps, attn, largest=True, mask_ratio=mask_ratio_h,
                                                         len_keep_other=ps, random_ratio=self.mask_ratio_hr,
                                                         select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:, :len_keep].squeeze() if self.select_inv else mask_ids[:,
                                                                                                   len_keep:].squeeze()

                len_keep, mask_ids = self.select_mask_fn(ps, attn, largest=True, mask_ratio=mask_ratio_h,
                                                         mask_ids_other=mask_ids, len_keep_other=ps,
                                                         cls_attn_topk_idx_other=cls_attn_topk_idx_other,
                                                         random_ratio=self.mask_ratio_hr, select_inv=self.select_inv)

        # recover the group features
        if self.mask_non_group_feat and is_group_feat is not None:
            len_keep = group_ps_len + len_keep
            mask_ids = torch.cat([group_indices.unsqueeze(0), mask_ids], dim=1)

        return len_keep, mask_ids

        # Modified by MAE@Meta

    def masking(self, x, ids_shuffle=None, len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _, ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask.squeeze(0), ids_restore

    def compute_attn_with_grad(self, x, label=None):
        with torch.no_grad():
            x = self.patch_to_emb(x)
            x = self.dp(x)

        x, attn, act = self.online_encoder(x, return_attn=True, return_act=True)

        if isinstance(attn, (list, tuple)):
            attn = torch.cat([attn[i].mean(dim=1).squeeze(0) for i in range(len(attn))], dim=0)
        else:
            attn = attn.mean(dim=0).squeeze(0)

        return attn


    @torch.no_grad()
    def forward_teacher(self, x, return_attn=False, label=None):

        x = self.patch_to_emb(x)
        x = self.dp(x)

        x_shortcut = x.clone()

        if self.backbone == 'dsmil':
            if return_attn:
                x, _, _, attn = self.online_encoder(x, return_attn=True, label=label)
                if self.test_merge:
                    attn = attn[0, :p]
            else:
                x, _, _ = self.online_encoder(x)
                attn = None
        else:
            if return_attn:
                # classifier = self.predictor if self.attn2score else None
                x, attn, act = self.online_encoder(x, return_attn=True, return_act=True)
                if self.test_merge:
                    if type(attn) in (list, tuple):
                        attn = [attn[i][0, :, :p] for i in range(len(attn))]
                    else:
                        attn = attn[0, :p]
            else:
                x = self.online_encoder(x)
                attn = None

            if self.attn2score:
                self.score_label = None if self.score_label == -1 else self.score_label
                if self.backbone == 'selfattn':
                    attn = get_cam_1d_trans(self.predictor, act, attn[0], self.score_label,
                                            self.online_encoder.attn_model.layer2.attn.to_out)
                else:
                    # attn = get_cam_1d(self.predictor,x_shortcut,attn,self.score_label)
                    attn = get_cam_1d(self.predictor, act, attn, self.score_label)
        ## softmax need to set dim=1
        return x, attn, self.predictor(x)

    def forward_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat, teacher_cls_feat.detach())
        else:
            cls_loss = 0.

        return cls_loss

    # for train loop
    def forward(self, x, attn=None, teacher_cls_feat=None, i=None, label=None, criterion=None, pred_correct=False,
                is_group_feat=None, relative_area=None):
        x = self.patch_to_emb(x)
        x = self.dp(x)

        bs, ps, _ = x.size()

        # get mask
        if self.select_mask and is_group_feat is not None:
            # print(attn.size())
            len_keep, mask_ids = self.get_mask(ps, i, attn, pred_correct=pred_correct, is_group_feat=is_group_feat,
                                               relative_area=relative_area)
            x, mask, _ = self.masking(x, mask_ids, len_keep)
        else:
            mask_ids, mask = None, None

        # x = self.merge(x)
        len_keep = x.size(1)

        if self.backbone == 'dsmil':
            # forward online network
            student_cls_feat, student_logit, inst_loss = self.online_encoder(x, label=label, criterion=criterion)

            logits_loss = criterion(student_logit.view(bs, -1), label)
            logits_loss = logits_loss * 0.5 + inst_loss * 0.5

            # cl loss
            cls_loss = self.forward_loss(student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat)

            return logits_loss, cls_loss, ps, len_keep

        else:
            # forward online network
            student_cls_feat = self.online_encoder(x)

            # prediction
            student_logit = self.predictor(student_cls_feat)

            # cl loss
            if self.cl_type == 'logits':
                cls_loss = self.forward_loss(student_cls_feat=student_logit, teacher_cls_feat=teacher_cls_feat)
            elif self.cl_type == 'feat':
                cls_loss = self.forward_loss(student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat)

            return student_logit, cls_loss, ps, len_keep

