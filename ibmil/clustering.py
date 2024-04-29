import enum
import re
from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
from torch.utils.data import DataLoader,Sampler, WeightedRandomSampler, RandomSampler
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset 
import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve
import random 
import torch.backends.cudnn as cudnn
import json
torch.multiprocessing.set_sharing_strategy('file_system')
import faiss
from utils import *
from dataloader import *

def seed_torch(seed=2021):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  


def preprocess_features(npdata, pca):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    assert npdata.dtype == np.float32

    if np.any(np.isnan(npdata)):
        raise Exception("nan occurs")
    if pca != -1:
        print("\nPCA from dim {} to dim {}".format(ndim, pca))
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(npdata)
        assert mat.is_trained
        npdata = mat.apply_py(npdata)
    if np.any(np.isnan(npdata)):
        percent = np.isnan(npdata).sum().item() / float(np.size(npdata)) * 100
        if percent > 0.1:
            raise Exception(
                "More than 0.1% nan occurs after pca, percent: {}%".format(
                    percent))
        else:
            npdata[np.isnan(npdata)] = 0.
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)

    npdata = npdata / (row_sums[:, np.newaxis] + 1e-10)

    return npdata

def run_kmeans(x, nmb_clusters, verbose=False, seed=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    if seed is not None:
        clus.seed = seed
    else:
        clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    return [int(n[0]) for n in I]

class Kmeans:

    def __init__(self, k, pca_dim=256):
        self.k = k
        self.pca_dim = pca_dim

    def cluster(self, feat, verbose=False, seed=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(feat, self.pca_dim)

        # cluster the data
        I = run_kmeans(xb, self.k, verbose, seed)
        self.labels = np.array(I)
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

    
def reduce(args, feats, k, fold):
    '''
    feats:bag feature tensor,[N,D]
    k: number of clusters
    shift: number of cov interpolation
    '''
    prototypes = []
    semantic_shifts = []
    feats = feats.cpu().numpy()

    kmeans = Kmeans(k=k, pca_dim=-1)
    kmeans.cluster(feats, seed=66)  # for reproducibility
    assignments = kmeans.labels.astype(np.int64)
    # compute the centroids for each cluster
    centroids = np.array([np.mean(feats[assignments == i], axis=0)
                          for i in range(k)])

    # compute covariance matrix for each cluster
    covariance = np.array([np.cov(feats[assignments == i].T)
                           for i in range(k)])

    os.makedirs(f'./{args.datasets}/{args.seed}', exist_ok=True)
    os.makedirs(f'./{args.datasets}/{args.seed}/{fold}', exist_ok=True)
    prototypes.append(centroids)
    prototypes = np.array(prototypes)
    prototypes =  prototypes.reshape(-1, 512)
    print(prototypes.shape)
    print(f'./{args.datasets}/{args.seed}/{fold}/train_bag_cls_agnostic_feats_proto_{k}.npy')
    np.save(f'./{args.datasets}/{args.seed}/{fold}/train_bag_cls_agnostic_feats_proto_{k}.npy', prototypes)

    del feats


def main():
    parser = argparse.ArgumentParser(description='Clutering for abmil/dsmil/transmil')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--model', default='abmil', type=str, help='MIL model [admil, dsmil]')
    parser.add_argument('--datasets', default='camelyon16', type=str, help='Dataset folder name')
    parser.add_argument('--dataset_root', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--load_path', default='./', type=str, help='load path for Stage 2')
    parser.add_argument('--seed', default=2021, type=int, help='random number [7]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--val_ratio', default=0., type=float, help='Automatic Mixed Precision Training')
    parser.add_argument('--persistence', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of cross validation fold [3]')
    # parser.add_argument('--dir', type=str,help='directory to save logs')
    #dsmil
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')

    args = parser.parse_args()
    # args = parser.parse_args(['--feats_size', '512','--num_classes','2', '--dataset','tcga_Img_nor'])
    '''
    ['--feats_size','512', '--num_classes','1', '--dataset','Camelyon16_Img_nor']
    ['--feats_size', '512','--num_classes','2', '--dataset','tcga_Img_nor']
    '''

    seed_torch(args.seed)


    # if args.dataset.startswith("tcga"):
    #     bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
    #     bags_path = pd.read_csv(bags_csv)
    #     train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
    #     test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]
    # elif args.dataset.startswith('Camelyon16'):
    #     # bags_csv = os.path.join('datasets', args.dataset, args.dataset+'_off.csv') #offical train test
    #     bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
    #     bags_path = pd.read_csv(bags_csv)
    #     train_path = bags_path.iloc[0:270, :]
    #     test_path = bags_path.iloc[270:, :]
            
    # trainset =  BagDataset(train_path, args)
    # train_loader = DataLoader(trainset,1, shuffle=True, num_workers=16)

    # --->划分数据集
    if args.datasets.lower() == 'camelyon16':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    elif args.datasets.lower() == 'tcga':
        label_path=os.path.join(args.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    # 当cv_fold == 1时，不使用交叉验证
    if args.cv_fold > 1:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio,True)


    for k in range(0, args.cv_fold):
        one_fold(args,k,train_p, train_l, test_p, test_l,val_p,val_l)

def one_fold(args,k,train_p, train_l, test_p, test_l,val_p,val_l):
    seed_torch(args.seed)
    # --->加载数据
    if args.datasets.lower() == 'camelyon16':

        train_set = C16Dataset(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence,is_train=True)
        test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence)
        if args.val_ratio != 0.:
            val_set = C16Dataset(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence)
        else:
            val_set = test_set
        # _f = open('c16_dataset.txt','a',encoding='utf-8')
        
        # return [acs,pre,rec,fs,auc,te_auc,te_fs]

        big_seed_list = 7784414403328510413
        generator = torch.Generator()
        generator.manual_seed(big_seed_list)  
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.num_workers,generator=generator)

    elif args.datasets.lower() == 'tcga':
        
        train_set = TCGADataset(train_p[k],train_l[k],-1,args.dataset_root,persistence=args.persistence,is_train=True)
        test_set = TCGADataset(test_p[k],test_l[k],-1,args.dataset_root,persistence=args.persistence)
        if args.val_ratio != 0.:
            val_set = TCGADataset(val_p[k],val_l[k],-1,args.dataset_root,persistence=args.persistence)
        else:
            val_set = test_set

        train_loader = DataLoader(train_set, batch_size=1, sampler=RandomSampler(train_set), num_workers=args.num_workers)
 
    if args.model == 'dsmil':
        import dsmil as mil
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    elif args.model == 'abmil':
        import abmil as mil
        milnet = mil.Dattention_ori(in_size=args.feats_size, out_dim=args.num_classes).cuda()
    elif args.model == 'transmil':
        import transmil as mil
        milnet = mil.TransMIL(input_size=args.feats_size, n_classes=args.num_classes).cuda()

    if not args.load_path.endswith('.pt'):
        _str = 'fold_{fold}_model_best_auc.pt'.format(fold=k)
        _load_path = os.path.join(args.load_path,_str)
    else:
        _load_path =args.load_path

    state_dict_weights = torch.load(_load_path) 


    if 'model' in state_dict_weights:
        model_weights = state_dict_weights['model']
        for key in model_weights.keys():
            print(key)
        
        # 定义预训练权重中的键到当前模型中的键的映射
        # key_map = {
        #     'patch_to_emb.0.weight': 'embedding.embed.0.weight',
        #     'patch_to_emb.0.bias': 'embedding.embed.0.bias',
        #     'online_encoder.attention.attention.0.weight': 'attention.0.weight',
        #     'online_encoder.attention.attention.2.weight': 'attention.2.weight',
        #     'predictor.weight': 'head.weight',
        #     'predictor.bias': 'head.bias'
        # }
        key_map = {
            'feature.0.weight': 'embedding.embed.0.weight',
            'feature.0.bias': 'embedding.embed.0.bias',
            'classifier.0.weight': 'head.weight',
            'classifier.0.bias': 'head.bias',   
        }

        # 根据映射更新权重中的键名
        mapped_weights = {}
        for key, value in model_weights.items():
            new_key = key_map.get(key, key)  # 如果键存在于映射中，则替换，否则使用原键
            mapped_weights[new_key] = value
        
        # 使用映射后的权重进行加载
        msg = milnet.load_state_dict(mapped_weights, strict=False)

    else:
        msg = milnet.load_state_dict(state_dict_weights, strict=False)
    print("***********loading init from {}*******************".format(_load_path))
    print(msg)
    milnet.eval()

    # forward
    feats_list = []
    for i,(bag_feats,bag_label) in enumerate(train_loader):
        with torch.no_grad():
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim  
            if args.model == 'abmil':
                bag_prediction, bag_feats = milnet(bag_feats)
            elif args.model == 'dsmil':
                ins_prediction, bag_prediction, attention, bag_feats= milnet(bag_feats)
            elif args.model == 'transmil':
                output = milnet(bag_feats)
                bag_prediction, bag_feats ,attention=  output['logits'], output["Bag_feature"], output["A"]
            
            feats_list.append(bag_feats.cpu())

    bag_tensor = torch.cat(feats_list,dim=0)
    bag_tensor_ag = bag_tensor.view(-1,512)
    reduce(args, bag_tensor_ag, 2, k)

if __name__ == '__main__':
    main()