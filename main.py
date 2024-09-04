import time
import torch
import wandb
import os
import numpy as np
from copy import deepcopy

import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler,SubsetRandomSampler
from torch.cuda.amp import GradScaler

from contextlib import suppress
from timm.utils import AverageMeter
from collections import OrderedDict

from dataloader import *
from modules import attmil,clam,mhim,dsmil,transmil,mean_max,dtfd,ibmil,sam
from utils import *
from engine import build_engine
from options import parse_args

def main(args):
    # set seed
    seed_torch(args.seed)

    # --->get dataset
    label_path = args.csv_path if args.csv_path else os.path.join(args.dataset_root,'label.csv')
    p, l = get_patient_label(label_path)
    index = [i for i in range(len(p))]
    random.shuffle(index)
    p = p[index]
    l = l[index]
    if args.cv_fold > 1:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)
        dataset = [train_p, train_l, test_p, test_l,val_p,val_l]
    
    # to be updated
    if args.datasets.lower().startswith('surv'):
        cindex, te_cindex = [],[]
        ckc_metric = [cindex]
        te_ckc_metric = [te_cindex]
    else:
        acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
        ckc_metric = [acs, pre, rec,fs,auc]
        te_ckc_metric = [te_auc,te_fs]

    if not args.no_log:
        print('Dataset: ' + args.datasets)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        args.fold_start = ckp['k']
        if len(ckp['ckc_metric']) == 6:
            acs, pre, rec,fs,auc,te_auc = ckp['ckc_metric']
        elif len(ckp['ckc_metric']) == 7:
            acs, pre, rec,fs,auc,te_auc,te_fs = ckp['ckc_metric']
        elif len(ckp['ckc_metric']) == 2:
            cindex, te_cindex = ckp['ckc_metric']
        else:
            acs, pre, rec,fs,auc = ckp['ckc_metric']

    for k in range(args.fold_start, args.cv_fold):
        if not args.no_log:
            print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric,te_ckc_metric = one_fold(args,k,ckc_metric,te_ckc_metric,dataset)

    if args.always_test:
        if args.wandb:
            wandb.log({
                "cross_val/te_auc_mean":np.mean(np.array(te_auc)),
                "cross_val/te_auc_std":np.std(np.array(te_auc)),
                "cross_val/te_f1_mean":np.mean(np.array(te_fs)),
                "cross_val/te_f1_std":np.std(np.array(te_fs)),
            })

    if args.wandb:
        wandb.log({
            "cross_val/acc_mean":np.mean(np.array(acs)),
            "cross_val/auc_mean":np.mean(np.array(auc)),
            "cross_val/f1_mean":np.mean(np.array(fs)),
            "cross_val/pre_mean":np.mean(np.array(pre)),
            "cross_val/recall_mean":np.mean(np.array(rec)),
            "cross_val/acc_std":np.std(np.array(acs)),
            "cross_val/auc_std":np.std(np.array(auc)),
            "cross_val/f1_std":np.std(np.array(fs)),
            "cross_val/pre_std":np.std(np.array(pre)),
            "cross_val/recall_std":np.std(np.array(rec)),
        })
    if not args.no_log:
        print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

def one_fold(args,k,ckc_metric,te_ckc_metric,dataset):
    # --->initiation
    seed_torch(args.seed)
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # --->load data
    train_p, train_l, test_p, test_l,val_p,val_l = dataset
    if args.datasets.lower() == 'camelyon16':
        if args.sam_mask and args.model == 'sam':
            train_set = C16Dataset_SAM(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
            test_set = C16Dataset_SAM(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
            if args.val_ratio != 0.:
                val_set = C16Dataset_SAM(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
            else:
                val_set = test_set
        else:
            train_set = C16Dataset(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
            test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
            if args.val_ratio != 0.:
                val_set = C16Dataset(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
            else:
                val_set = test_set
    elif args.datasets.lower() == 'tcga':
        if args.sam_mask and args.model == 'sam':
            train_set= TCGADataset_SAM(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True,_type=args.tcga_sub)
            test_set = TCGADataset_SAM(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
            if args.val_ratio != 0.:
                val_set = TCGADataset_SAM(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
            else:
                val_set = test_set
        else:
            train_set = TCGADataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True,_type=args.tcga_sub)
            test_set = TCGADataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
            if args.val_ratio != 0.:
                val_set = TCGADataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
            else:
                val_set = test_set

    if args.fix_loader_random:
        # generated by int(torch.empty((), dtype=torch.int64).random_().item())
        big_seed_list = 7784414403328510413
        generator = torch.Generator()
        generator.manual_seed(big_seed_list)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    mm_sche = None
    if not args.teacher_init.endswith('.pt'):
        _str = 'fold_{fold}_model_best_auc.pt'.format(fold=k)
        _teacher_init = os.path.join(args.teacher_init,_str)
    else:
        _teacher_init =args.teacher_init

    # --->bulid networks
    if args.model == 'sam':
        if args.mrh_sche:
            mrh_sche = cosine_scheduler(args.mask_ratio_h,0.,epochs=args.num_epoch,niter_per_ep=len(train_loader))
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'input_dim':args.input_dim,
            'mask_ratio' : args.mask_ratio,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
            'attn2score': args.attn2score,
            'select_mask': args.select_mask,
            'sam_mask': args.sam_mask,
            'sigmoid_k': args.sigmoid_k,
            'sigmoid_A0': args.sigmoid_A0,
            'mask_non_group_feat': args.mask_non_group_feat,
            'mask_by_seg_area': args.mask_by_seg_area
        }
        
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.num_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)
        model = sam.sam_mil(**model_params).to(device)

    elif args.model == 'mhim':
        if args.mrh_sche:
            mrh_sche = cosine_scheduler(args.mask_ratio_h, 0., epochs=args.num_epoch, niter_per_ep=len(train_loader))
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'input_dim': args.input_dim,
            'mask_ratio': args.mask_ratio,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
            'attn2score': args.attn2score,
            'merge_enable': args.merge_enable,
            'merge_k': args.merge_k,
            'merge_mm': args.merge_mm,
            'merge_ratio': args.merge_ratio,
            'merge_test': args.merge_test,
            'merge_mask_type': args.merge_mask_type
        }

        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm, args.mm_final, epochs=args.num_epoch, niter_per_ep=len(train_loader),
                                       start_warmup_value=1.)

        model = mhim.MHIM(**model_params).to(device)
    elif args.model == 'pure':
        model = mhim.MHIM(input_dim=args.input_dim,select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)
    elif args.model == 'attmil':
        model = attmil.DAttention(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'gattmil':
        model = attmil.AttentionGated(input_dim=args.input_dim,dropout=args.dropout).to(device)
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        model = clam.CLAM_SB(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'clam_mb':
        model = clam.CLAM_MB(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'transmil':
        model = transmil.TransMIL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'dsmil':
        model = dsmil.MILNet(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5
        state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        info = model.load_state_dict(state_dict_weights, strict=False)
        if not args.no_log:
            print(info)
    elif args.model == 'dtfd':
        # group: 5,8
        # distill: MinMaxS, AFS
        model = dtfd.DTFD(lr=args.lr, weight_decay=args.weight_decay, steps=args.num_epoch, input_dim=args.input_dim, n_classes=args.n_classes,criterion=NLLSurvLoss(alpha=0.0),group=args.pseudo_bags,distill=args.distill_type).to(device)
    elif args.model == 'ibmil':
        if not args.confounder_path.endswith('.npy'):
            _confounder_path = os.path.join(args.confounder_path,str(k),'train_bag_cls_agnostic_feats_proto_'+str(args.confounder_k)+'.npy')
        else:
            _confounder_path =args.confounder_path
        model = ibmil.Dattention_ori(out_dim=args.n_classes,dropout=args.dropout,in_size=args.input_dim,confounder_path=_confounder_path).to(device)
    elif args.model == 'meanmil':
        model = mean_max.MeanMIL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'maxmil':
        model = mean_max.MaxMIL(input_dim=args.input_dim,n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    #### MHIM Init  ####
    if args.init_stu_type != 'none':
        if not args.no_log:
            print('######### Model Initializing.....')
        pre_dict = torch.load(_teacher_init)
        if 'model' in pre_dict:
            pre_dict = pre_dict['model']
        new_state_dict ={}
        if args.init_stu_type == 'fc':
        # only patch_to_emb
            for _k,v in pre_dict.items():
                _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                new_state_dict[_k]=v
            info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
        else:
        # init all
            info = model.load_state_dict(pre_dict,strict=False)
        if not args.no_log:
            print(info)
    # teacher model
    if args.model == 'mhim':
        model_tea = deepcopy(model)
        if not args.no_tea_init and args.tea_type != 'same':
            if not args.no_log:
                print('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(_teacher_init)
                if 'model' in pre_dict:
                    pre_dict = pre_dict['model']
                info = model_tea.load_state_dict(pre_dict,strict=False)
                if not args.no_log:
                    print(info)
            except:
                if not args.no_log:
                    print('########## Init Error')
        if args.tea_type == 'same':
            model_tea = model
        model_tea.merge_test = False
    elif args.model == 'sam':
        model_tea = deepcopy(model)
        model_tea.test_merge = False
    else:
        model_tea = None

    # build criterion
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "nll_surv":
        criterion = NLLSurvLoss(alpha=0.0)
    # build optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    # build scheduler
    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None
    # build early stopping
    if args.early_stopping:
        if args.datasets.lower().startswith('surv'):
            patience,stop_epoch = 10,30
        elif 'camelyon' in args.datasets:
            patience,stop_epoch = 30,args.max_epoch
        elif args.datasets == 'tcga':
            patience,stop_epoch = 20,70
        early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None
    # metric
    best_ckc_metric = [0. for i in range(len(ckc_metric))]
    best_ckc_metric_te = [0. for i in range(len(te_ckc_metric))]
    best_ckc_metric_te_tea = [0. for i in range(len(te_ckc_metric))]
    epoch_start,opt_thr,opt_main_tea = 0,0,0

    if args.fix_train_random:
        seed_torch(args.seed)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        epoch_start = ckp['epoch']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['lr_sche'])
        early_stopping.load_state_dict(ckp['early_stop'])
        opt_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = ckp['val_best_metric']
        opt_te_auc = ckp['te_best_metric'][0]
        if len(ckp['te_best_metric']) > 1:
            opt_te_fs = ckp['te_best_metric'][1]
        opt_te_tea_auc,opt_te_tea_fs = ckp['te_best_metric'][2:4]
        np.random.set_state(ckp['random']['np'])
        torch.random.set_rng_state(ckp['random']['torch'])
        random.setstate(ckp['random']['py'])
        if args.fix_loader_random:
            train_loader.sampler.generator.set_state(ckp['random']['loader'])
        args.auto_resume = False

    # build engine
    train_loop,val_loop,test = build_engine(args)

    # train loop
    train_time_meter = AverageMeter()
    for epoch in range(epoch_start, args.num_epoch):
        train_loss,start,end = train_loop(args,model,model_tea,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch)
        train_time_meter.update(end-start)
        stop,_metric_val, rowd_val,test_loss, threshold_optimal = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch,model_tea)

        if model_tea is not None:
            _,_metric_tea, rowd,test_loss_tea,_ = val_loop(args,model_tea,val_loader,device,criterion,None,epoch,model_tea)
            if args.wandb:
                rowd = OrderedDict([ (str(k)+'-fold/val_tea_'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if _metric_tea[0] > opt_main_tea:
                opt_main_tea = _metric_tea[0]
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_main_tea",opt_main_tea)
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)

        _te_metric = [0.,0.]
        if args.always_test:

            _te_metric,rowd,_te_test_loss_log = test(args,model,test_loader,device,criterion,model_tea)
            
            if args.wandb:
                rowd = OrderedDict([ (str(k)+'-fold/te_'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if _te_metric[0] > best_ckc_metric_te[0]:
                best_ckc_metric_te = _te_metric[0],_te_metric[1]
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_te_main",_te_metric[0]),
                        ("best_te_sub",_te_metric[1])
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)
            
            if model_tea is not None:
                _te_tea_metric,rowd,_te_tea_test_loss_log = test(args,model_tea,test_loader,device,criterion,model_tea)
            
                if args.wandb:
                    rowd = OrderedDict([ (str(k)+'-fold/te_tea_'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)

                if _te_tea_metric[0] > best_ckc_metric_te_tea[0]:
                    best_ckc_metric_te_tea = _te_tea_metric[0],_te_tea_metric[1]
                    if args.wandb:
                        rowd = OrderedDict([
                            ("best_te_tea_main",_te_tea_metric[0]),
                            ("best_te_tea_sub",_te_tea_metric[1])
                        ])
                        rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                        wandb.log(rowd)
        if not args.no_log:
            if args.datasets.lower().startswith('surv'):
                print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, c-index: %.3f, time: %.3f(%.3f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[0], train_time_meter.val,train_time_meter.avg))
            else:
                print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % (epoch+1, args.num_epoch, train_loss, test_loss, _metric_val[2], _metric_val[0], _metric_val[3], _metric_val[4], _metric_val[1], train_time_meter.val,train_time_meter.avg))

        if args.wandb:
            rowd_val['epoch'] = epoch
                
            rowd = OrderedDict([ (str(k)+'-fold/val_'+_k,_v) for _k, _v in rowd_val.items()])
            wandb.log(rowd)

        if _metric_val[0] > best_ckc_metric[0] and epoch >= args.save_best_model_stage*args.num_epoch:
            best_ckc_metric = _metric_val+[epoch]
            best_rowd = rowd_val
            opt_thr = threshold_optimal

            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                    'teacher': model_tea.state_dict() if model_tea is not None else None,
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        if epoch == 0:
            best_rowd = rowd_val
        if args.wandb:
            rowd = OrderedDict([ (str(k)+'-fold/val_best_'+_k,_v) for _k, _v in best_rowd.items()])
            wandb.log(rowd)
        
        # save checkpoint
        random_state = {
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'py': random.getstate(),
            'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
        }     
        ckp = {
            'model': model.state_dict(),
            'lr_sche': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'k': k,
            'early_stop': early_stopping.state_dict(),
            'random': random_state,
            'ckc_metric': _metric_val+_te_metric,
            'val_best_metric': best_ckc_metric,
            'te_best_metric': best_ckc_metric_te+best_ckc_metric_te_tea,
            'wandb_id': wandb.run.id if args.wandb else '',
        }
        if not args.no_log:
            torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))

        if stop:
            break
    
    # test
    if not args.no_log:
        best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        info = model.load_state_dict(best_std['model'])
        print(info)
        if model_tea is not None and best_std['teacher'] is not None:
            info = model_tea.load_state_dict(best_std['teacher'])
            print(info)

    metric_test,rowd,test_loss_log = test(args,model,test_loader,device,criterion,model_tea,opt_thr)
    
    if args.wandb:
        rowd = OrderedDict([ ('test_'+_k,_v) for _k, _v in rowd.items()])
        wandb.log(rowd)
    # if not args.no_log:
    #     print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (opt_ac,opt_auc,opt_pre,opt_re,opt_fs))
    
    [ckc_metric[i].append(metric_test[i]) for i,_ in enumerate(ckc_metric)]

    if args.always_test:
        [te_ckc_metric[i].append(best_ckc_metric_te[i]) for i,_ in enumerate(te_ckc_metric)]
        
    return ckc_metric,te_ckc_metric

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(os.path.join(args.model_path,args.project)):
        os.mkdir(os.path.join(args.model_path,args.project))
    args.model_path = os.path.join(args.model_path,args.project,args.title)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # survival prediction
    if args.datasets.lower().startswith('surv'):
        args.n_classes = 4

    if args.model == 'pure':
        args.cl_alpha=0.
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'clam_mb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'dsmil':
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5

    if args.datasets == 'camelyon16':
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets == 'tcga':
        args.num_workers = 0
        args.always_test = True

    if args.wandb:
        if args.auto_resume:
            ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
            wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path),id=ckp['wandb_id'],resume='must')
        else:
            wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path))
        
    print(args)

    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    main(args=args)