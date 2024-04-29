# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class Attention(nn.Module):
#     def __init__(self, in_size, out_size, confounder_path=False, confounder_learn=False, \
#         confounder_dim=128, confounder_merge='cat'):
#         super(Attention, self).__init__()
#         self.L = in_size
#         self.D = in_size
#         self.K = 1
#         self.confounder_merge = confounder_merge
#         assert confounder_merge in ['cat', 'add', 'sub']
#         # self.feature_extractor_part1 = nn.Sequential(
#         #     nn.Conv2d(1, 20, kernel_size=5),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(2, stride=2),
#         #     nn.Conv2d(20, 50, kernel_size=5),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(2, stride=2)
#         # )

#         # self.feature_extractor_part2 = nn.Sequential(
#         #     nn.Linear(50 * 4 * 4, self.L),
#         #     nn.ReLU(),
#         # )

#         # self.attention_1 = nn.Sequential(
#         #     nn.Linear(self.L, self.D),
#         #     nn.Tanh(),
            
#         # )
#         # self.attention_1 = nn.Identity()
#         # self.attention_2 = nn.Linear(self.D, self.K)
#         self.attention = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh(),
#             nn.Linear(self.D, self.K)
#         )
#         self.classifier =  nn.Linear(self.L*self.K, out_size)
#         self.confounder_path=None
#         if confounder_path: 
#             print('deconfounding')
#             self.confounder_path = confounder_path
#             conf_list = []
#             for i in confounder_path:
#                 conf_list.append(torch.from_numpy(np.load(i)).view(-1,in_size).float())
#             conf_tensor = torch.cat(conf_list, 0) 
#             conf_tensor_dim = conf_tensor.shape[-1]
#             if confounder_learn:
#                 self.confounder_feat = nn.Parameter(conf_tensor, requires_grad=True)
#             else:
#                 self.register_buffer("confounder_feat",conf_tensor)
#             joint_space_dim = confounder_dim
#             dropout_v = 0.5
#             # self.confounder_W_q = nn.Linear(in_size, joint_space_dim)
#             # self.confounder_W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
#             self.W_q = nn.Linear(in_size, joint_space_dim)
#             self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
#             if confounder_merge == 'cat':
#                 self.classifier =  nn.Linear(self.L*self.K+conf_tensor_dim, out_size)
#             elif confounder_merge == 'add' or 'sub':
#                 self.classifier =  nn.Linear(self.L*self.K, out_size)
#             self.dropout = nn.Dropout(dropout_v)

#     def forward(self, x):
#         # x = x.squeeze(0)

#         # H = self.feature_extractor_part1(x)
#         # H = H.view(-1, 50 * 4 * 4)
#         # H = self.feature_extractor_part2(H)  # NxL

#         # A = self.attention_1(x)  
#         # A = self.attention_2(A)  # NxK
#         A = self.attention(x)  # NxK
#         A = torch.transpose(A, 1, 0)  # KxN
#         A = F.softmax(A, dim=1)  # softmax over N
#         # print('norm')
#         # A = F.softmax(A/ torch.sqrt(torch.tensor(x.shape[1])), dim=1)  # For Vis

#         M = torch.mm(A, x)  # KxL
#         if self.confounder_path:
#             device = M.device
#             # bag_q = self.confounder_W_q(M)
#             # conf_k = self.confounder_W_k(self.confounder_feat)
#             bag_q = self.W_q(M)
#             conf_k = self.W_k(self.confounder_feat)
#             deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
#             deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
#             conf_feats = torch.mm(deconf_A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
#             if self.confounder_merge == 'cat':
#                 M = torch.cat((M,conf_feats),dim=1)
#             elif self.confounder_merge == 'add':
#                 M = M + conf_feats
#             elif self.confounder_merge == 'sub':
#                 M = M - conf_feats
#         Y_prob = self.classifier(M)
#         Y_hat = torch.ge(Y_prob, 0.5).float()
#         if self.confounder_path:
#             return Y_prob, M, deconf_A
#         else:
#             return Y_prob, M, A

#     # AUXILIARY METHODS
#     def calculate_classification_error(self, X, Y):
#         Y = Y.float()
#         _, Y_hat, _ = self.forward(X)
#         error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

#         return error, Y_hat

#     def calculate_objective(self, X, Y):
#         Y = Y.float()
#         Y_prob, _, A = self.forward(X)
#         Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
#         neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

#         return neg_log_likelihood, A

# class GatedAttention(nn.Module):
#     def __init__(self):
#         super(GatedAttention, self).__init__()
#         self.L = 500
#         self.D = 128
#         self.K = 1

#         self.feature_extractor_part1 = nn.Sequential(
#             nn.Conv2d(1, 20, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2),
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, stride=2)
#         )

#         self.feature_extractor_part2 = nn.Sequential(
#             nn.Linear(50 * 4 * 4, self.L),
#             nn.ReLU(),
#         )

#         self.attention_V = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Tanh()
#         )

#         self.attention_U = nn.Sequential(
#             nn.Linear(self.L, self.D),
#             nn.Sigmoid()
#         )

#         self.attention_weights = nn.Linear(self.D, self.K)

#         self.classifier = nn.Sequential(
#             nn.Linear(self.L*self.K, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x.squeeze(0)

#         H = self.feature_extractor_part1(x)
#         H = H.view(-1, 50 * 4 * 4)
#         H = self.feature_extractor_part2(H)  # NxL

#         A_V = self.attention_V(H)  # NxD
#         A_U = self.attention_U(H)  # NxD
#         A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
#         A = torch.transpose(A, 1, 0)  # KxN
#         A = F.softmax(A, dim=1)  # softmax over N

#         M = torch.mm(A, H)  # KxL

#         Y_prob = self.classifier(M)
#         Y_hat = torch.ge(Y_prob, 0.5).float()

#         return Y_prob, Y_hat, A

#     # AUXILIARY METHODS
#     def calculate_classification_error(self, X, Y):
#         Y = Y.float()
#         _, Y_hat, _ = self.forward(X)
#         error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

#         return error, Y_hat

#     def calculate_objective(self, X, Y):
#         Y = Y.float()
#         Y_prob, _, A = self.forward(X)
#         Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
#         neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

#         return neg_log_likelihood, A

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import sys
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

class FCLayer(nn.Module):
    def __init__(self, dropout=0.25,act='relu',in_size=1024):
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

        self.embed = nn.Sequential(*self.embed)

    def forward(self, feats):
        feats = self.embed(feats)
        return feats

class Dattention_ori(nn.Module):
    def __init__(self,out_dim=2,in_size=1024,dropout=0.25,confounder_path=False):
        super(Dattention_ori,self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.embedding = FCLayer(in_size=in_size,dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.head = nn.Linear(512,out_dim)
        self.confounder_path = confounder_path
        if confounder_path: 
            print('deconfounding')
            self.confounder_path = confounder_path
            conf_list = []
            for i in confounder_path:
                conf_list.append(torch.from_numpy(np.load(i)).view(-1,in_size).float())
            conf_tensor = torch.cat(conf_list, 0) 
            conf_tensor_dim = conf_tensor.shape[-1]
            self.register_buffer("confounder_feat",conf_tensor)
            joint_space_dim = 128
            dropout_v = 0.5
            # self.confounder_W_q = nn.Linear(in_size, joint_space_dim)
            # self.confounder_W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.W_q = nn.Linear(in_size, joint_space_dim)
            self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.classifier =  nn.Linear(self.L*self.K+conf_tensor_dim, out_dim)
            self.dropout = nn.Dropout(dropout_v)

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

        return self.head(x),x

if __name__ == "__main__":
    x=torch.rand(5,3,64,64).cuda()

