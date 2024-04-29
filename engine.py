import time
import wandb

from torch.nn.functional import one_hot
from timm.models import  model_parameters
from timm.utils import AverageMeter,dispatch_clip_grad
from collections import OrderedDict
from sksurv.metrics import concordance_index_censored

from utils import *

def build_engine(args):
    return train_loop,val_loop,test

def train_loop(args,model,model_tea,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch):
    start = time.time()
    loss_cls_meter = AverageMeter()
    loss_cl_meter = AverageMeter()
    loss_bag_meter = AverageMeter()
    loss_consistency_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()
    mm_meter = AverageMeter()
    train_loss_log = 0.
    logit_loss = None
    model.train()
    if model_tea is not None:
        model_tea.train()

    for i, data in enumerate(loader):
        optimizer.zero_grad()

        if isinstance(data[0], (list, tuple)):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
            bag = data[0]
            batch_size = data[0][0].size(0)
        else:
            bag = data[0].to(device)  # b*n*1024
            batch_size = bag.size(0)

        label = data[1].to(device)
        if args.model == 'sam':
            is_group_feat = data[2].to(device)
            relative_area = data[3].to(device)

        logit_loss = None
        bag_loss = 0.
        consistency_loss = 0.

        with amp_autocast():
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)

            if args.model == 'sam':
                if model_tea is not None:
                    cls_tea, attn, pred = model_tea.forward_teacher(bag, return_attn=True, label=label)
                else:
                    attn, cls_tea, pre_correct = None, None, False
                cls_tea = None if args.cl_alpha == 0. else cls_tea
                cls_tea = pred if args.cl_type == 'logits' else cls_tea
                pred_correct = pred[0] == label[0] if args.correct_label_filter else False

                train_logits, cls_loss, patch_num, keep_num = model(bag, attn, cls_tea, i=epoch * len(loader) + i,
                                                                    pred_correct=pred_correct,
                                                                    is_group_feat=is_group_feat,
                                                                    relative_area=relative_area)
            elif args.model == 'mhim':
                if model_tea is not None:
                    cls_tea,attn = model_tea.forward_teacher(bag)
                else:
                    attn,cls_tea = None,None

                cls_tea = None if args.cl_alpha == 0. else cls_tea

                if args.baseline == 'dsmil':
                    logits, cls_loss,patch_num,keep_num = model(bag,attn,cls_tea[0],i=epoch*len(loader)+i)
                    logit_loss = 0.5*criterion(logits[0].view(batch_size,-1),label) + 0.5*criterion(logits[1].view(batch_size,-1),label)
                else:
                    logits, cls_loss,patch_num,keep_num = model(bag,attn,cls_tea,i=epoch*len(loader)+i)

            elif args.model == 'pure':
                if args.baseline == 'dsmil':
                    logits, cls_loss,patch_num,keep_num = model.pure(bag)
                    logit_loss = 0.5*criterion(logits[0].view(batch_size,-1),label) + 0.5*criterion(logits[1].view(batch_size,-1),label)
                else:
                    logits, cls_loss,patch_num,keep_num = model.pure(bag)
            elif args.model in ('clam_sb','clam_mb','dsmil'):
                logits,cls_loss,patch_num = model(bag,label,criterion)
                keep_num = patch_num
            else:
                logits = model(bag)
                cls_loss,patch_num,keep_num = 0.,0.,0.

            if args.model == 'sam' and args.group_alpha > 0 and args.num_group > 1:
                num_groups = args.num_group

                # generate random indices
                indices = torch.randperm(bag.size(1))  # use the dimension of bag

                # shuffle bag data
                shuffled_bag = bag[:, indices, :]

                # calculate the size of each group
                group_size = shuffled_bag.size(1) // num_groups

                # group the data
                bag_groups = [shuffled_bag[:, i * group_size:(i + 1) * group_size, :] for i in range(num_groups)]

                # if the total data cannot be divided by num_groups, add the remaining data to the last group
                if shuffled_bag.size(1) % num_groups != 0:
                    extra_bags = shuffled_bag[:, num_groups * group_size:, :]
                    bag_groups[-1] = torch.cat((bag_groups[-1], extra_bags), dim=1)

                # shuffle label data accordingly
                label_groups = [label for _ in range(num_groups)]

                group_results = []


                for group_idx, (group_bag, group_label) in enumerate(zip(bag_groups, label_groups)):
                    if model_tea is not None:
                        cls_tea, attn, pred = model_tea.forward_teacher(group_bag, return_attn=True, label=group_label)
                    else:
                        attn, cls_tea, pre_correct = None, None, False

                    cls_tea = None if args.cl_alpha == 0 else cls_tea
                    cls_tea = pred if args.cl_type == 'logits' else cls_tea
                    pred_correct = pred[0] == group_label[0] if args.correct_label_filter else False

                    train_logits, cls_loss, patch_num, keep_num = model(group_bag, attn, cls_tea,
                                                                        i=epoch * len(loader),
                                                                        pred_correct=pred_correct)

                    # save the results
                    group_results.append({
                        "train_logits": train_logits,
                        "cls_loss": cls_loss,
                        "patch_num": patch_num,
                        "keep_num": keep_num,
                        "pred": pred,
                        "pred_correct": pred_correct
                    })

                # initialize the total loss
                total_loss = 0.0

                # loop through the results of each group
                for (group_result, label_group) in zip(group_results, label_groups):
                    train_logits = group_result['train_logits']
                    batch_size = label_group.size(0)

                    if args.loss == 'ce':
                        logit_loss = criterion(train_logits.view(batch_size, -1), label_group)
                    elif args.loss == 'bce':
                        logit_loss = criterion(train_logits.view(batch_size, -1), one_hot(label_group, num_classes=2))

                    total_loss += logit_loss

                # calculate the average bag loss
                bag_loss = total_loss / len(group_results)

                if args.consistency_alpha > 0.:
                    global_attn = model_tea.compute_attn_with_grad(bag, label=label)
                    consistency_loss = calculate_consistency_loss(global_attn, relative_area, args.con_batch_size)
                else:
                    consistency_loss = 0.

                
            if logit_loss is None:
                if args.loss == 'ce':
                    logit_loss = criterion(logits.view(batch_size,-1),label)
                elif args.loss == 'bce':
                    logit_loss = criterion(logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))

        if args.model == 'sam' and args.group_alpha > 0. and args.num_group > 1:
            train_loss = args.cls_alpha * logit_loss + cls_loss * args.cl_alpha + bag_loss * args.group_alpha + consistency_loss * args.consistency_alpha
        else:
            train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha

        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            train_loss.backward()
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
            if args.model == 'mhim':
                if mm_sche is not None:
                    mm = mm_sche[epoch*len(loader)+i]
                else:
                    mm = args.mm
                if model_tea is not None:
                    if args.tea_type == 'same':
                        pass
                    else:
                        ema_update(model,model_tea,mm)
            else:
                mm = 0.

        loss_cls_meter.update(logit_loss,1)
        loss_cl_meter.update(cls_loss,1)
        loss_bag_meter.update(bag_loss,1)
        loss_consistency_meter.update(consistency_loss,1)
        patch_num_meter.update(patch_num,1)
        keep_num_meter.update(keep_num,1)
        mm_meter.update(mm,1)

        if i % args.log_iter == 0 or i == len(loader)-1:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('cls_loss',loss_cls_meter.avg),
                ('lr',lr),
                ('cl_loss',loss_cl_meter.avg),
                ('bag_loss',loss_bag_meter.avg),
                ('consistency_loss',loss_consistency_meter.avg),
                ('patch_num',patch_num_meter.avg),
                ('keep_num',keep_num_meter.avg),
                ('mm',mm_meter.avg),
            ])
            if not args.no_log:
                print(
                    '[{}/{}] logit_loss:{}, cls_loss:{}, bag_loss{}, consistency_loss{}, patch_num:{}, keep_num:{}'.format(
                        i, len(loader) - 1, loss_cls_meter.avg, loss_cl_meter.avg, loss_bag_meter.avg,
                        loss_consistency_meter.avg, patch_num_meter.avg, keep_num_meter.avg))
            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            if args.wandb:
                wandb.log(rowd)

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log,start,end

def val_loop(args,model,loader,device,criterion,early_stopping,epoch,model_tea=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    loss_cls_meter = AverageMeter()
    bag_logit, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
                if args.baseline == 'dsmil':
                    test_logits = test_logits[0]
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))) or (args.model == 'pure' and args.backbone == 'dsmil'):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label.view(batch_size,-1).float())
                    
                    bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            loss_cls_meter.update(test_loss,1)
    
    # save the log file
    accuracy, auc_value, precision, recall, fscore, threshold_optimal = five_scores(bag_labels, bag_logit)
    
    # early stop
    if early_stopping is not None:
        early_stopping(epoch,-auc_value,model)
        stop = early_stopping.early_stop
    else:
        stop = False
    
    rowd = OrderedDict([
                ("acc",accuracy),
                ("precision",precision),
                ("recall",recall),
                ("fscore",fscore),
                ("auc",auc_value),
                ("loss",loss_cls_meter.avg),
            ])

    return stop,[auc_value,fscore,accuracy, precision, recall],rowd,loss_cls_meter.avg, threshold_optimal

def test(args,model,loader,device,criterion,model_tea=None,opt_thr=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())
                
            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
                if args.baseline == 'dsmil':
                    test_logits = test_logits[0]
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))) or (args.model == 'pure' and args.backbone == 'dsmil'):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if batch_size > 1:
                        bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label.view(1,-1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    # save the log file
    # cal the best thr with val set
    opt_thr = opt_thr if args.best_thr_val else None
    accuracy, auc_value, precision, recall, fscore, _ = five_scores(bag_labels, bag_logit,threshold_optimal=opt_thr)
    test_loss_log = test_loss_log/len(loader)

    rowd = OrderedDict([
                ("acc",accuracy),
                ("precision",precision),
                ("recall",recall),
                ("fscore",fscore),
                ("auc",auc_value),
                ("loss",test_loss_log),
            ])

    return [auc_value,fscore,accuracy,  precision, recall],rowd,test_loss_log

############# Survival Prediction ###################

## To be updated

