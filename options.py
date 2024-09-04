import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Computional Pathology Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
    parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
    parser.add_argument('--csv_path', default=None, type=str, help='Dataset CSV path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    parser.add_argument('--tcga_sub', default='nsclc', type=str, help='[nsclc,brca]')

    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss, defualt nll_surv in survival prediction [ce, bce, nll_surv]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='sam', type=str, help='Model name')
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')
    parser.add_argument('--best_thr_val', action='store_true', help='Cal the best thr with val set in the test phase. Thanks Weiyi Wu!')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--baseline', default='selfattn', type=str, help='Baselin model [attn,selfattn]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # dtfd
    parser.add_argument('--pseudo_bags', default=8, type=int, help='Number of pseudo_bag') 
    parser.add_argument('--distill_type', default='MaxMinS', type=str, help='Type of distill')

    # IBMIL
    parser.add_argument('--confounder_k', default=2, type=int, help='number if confounder')
    parser.add_argument('--confounder_path', default=None, type=str, help='path of confounder')

    # MHIM
    # Mask ratio
    parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
    parser.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio')
    parser.add_argument('--mask_ratio_h', default=0., type=float, help='High attention mask ratio')
    parser.add_argument('--mask_ratio_hr', default=1., type=float, help='Randomly high attention mask ratio')
    parser.add_argument('--attn2score', action='store_true', help='cl loss alpha')
    parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
    parser.add_argument('--attn_layer', default=0, type=int)
    # Siamese framework
    parser.add_argument('--cl_alpha', default=0., type=float, help='Auxiliary loss alpha')
    parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
    parser.add_argument('--teacher_init', default='none', type=str, help='Path to initial teacher model')
    parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
    parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]')
    parser.add_argument('--tea_type', default='none', type=str, help='[none,same]')
    parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
    parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')
    # Merge
    parser.add_argument('--merge_enable', action='store_true',  help='Enable recycle')
    parser.add_argument('--merge_k', default=1, type=int, help='mask ratio')
    parser.add_argument('--merge_ratio', default=0.2, type=float, help='mask ratio')
    parser.add_argument('--merge_mm', default=0.9998, type=float, help='ema mm of global query')
    parser.add_argument('--merge_mask_type', default='random', type=str, help='mask type')
    parser.add_argument('--merge_test', action='store_true',  help='cl loss alpha')

    ## sam_mil
    parser.add_argument('--sam_mask', action='store_true', help='Enable SAM mask')
    parser.add_argument('--select_mask', action='store_true',  help='Enable select_mask')
    parser.add_argument('--sigmoid_k', default=0.0005, type=float, help='Adjustable sigmoid k')
    parser.add_argument('--sigmoid_A0', default=5000, type=float, help='Adjustable sigmoid A0')
    parser.add_argument('--mask_non_group_feat', action='store_true', help='Mask non-group feature')
    parser.add_argument('--mask_by_seg_area', action='store_true', help='Group mask by seg area')
    parser.add_argument('--num_group', default=3, type=int, help='number of pseudo-bags')
    parser.add_argument('--split_bag', action='store_true', help='Enable split bag')
    parser.add_argument('--group_alpha', default=0., type=float, help='alpha of group loss')
    parser.add_argument('--consistency_alpha', default=0., type=float, help='alpha of consistency loss')
    parser.add_argument("--con_batch_size", default=2048, type=int, help='batch size of consistency compute')

    # Misc
    parser.add_argument('--title', default='default', type=str, help='Title of exp')
    parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--model_path', type=str, help='Output path')

    return parser.parse_args()