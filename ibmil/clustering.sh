CUDA_VISIBLE_DEVICES=0 python3 clustering.py --num_classes 2 --datasets tcga --dataset_root /raid/Data/zhangyi/mil/Survival/blca/features/pt_files/resnet50/ --feats_size 1024 --model abmil --load_path /raid/Data/zhangyi/output/mil/abmil/ --cv_fold 5


CUDA_VISIBLE_DEVICES=0 python3 IBMIL-main/clustering-fh.py --num_classes 2 --datasets tcga --dataset_root /workspace/dataset/camelyon_all/plip_bioseg/ --feats_size 512 --model abmil --load_path /workspace/code/IBMIL-main/Models/plip_abmil/ --cv_fold 5



CUDA_VISIBLE_DEVICES=0 python3 main.py --project=pami_surv --datasets=surv_blca --dataset_root=/raid/Data/zhangyi/mil/Survival/blca/features/pt_files/resnet50/ --csv_path=./dataset_csv/label_blca_surv.csv --model_path=/raid/Data/zhangyi/output/mil --cv_fold=5 --val_ratio=0 --model=ibmil --loss=nll_surv --weight_decay=1e-5 --lr=2e-4 --seed=2021 --title=blca_ibmil_seed2021_k2_gated --opt=adam --lr_sche=cosine --num_workers=2 --input_dim=1024 --wandb --confounder_path=/home/zhangyi/tangwenhao/mil_new/datasets_deconf/BLCA/resnet50 --confounder_k=2
CUDA_VISIBLE_DEVICES=0 python3 main.py --project=pami_surv --datasets=surv_luad --dataset_root=/raid/Data/zhangyi/mil/Survival/luad/features/pt_files/resnet50/ --csv_path=./dataset_csv/label_luad_surv.csv --model_path=/raid/Data/zhangyi/output/mil --cv_fold=5 --val_ratio=0 --model=ibmil --loss=nll_surv --weight_decay=1e-5 --lr=2e-4 --seed=2021 --title=luad_ibmil_seed2021_k2_gated --opt=adam --lr_sche=cosine --num_workers=2 --input_dim=1024 --wandb --confounder_path=/home/zhangyi/tangwenhao/mil_new/datasets_deconf/LUAD/resnet50 --confounder_k=2
CUDA_VISIBLE_DEVICES=0 python3 main.py --project=pami_surv --datasets=surv_lusc --dataset_root=/raid/Data/zhangyi/mil/Survival/lusc/features/pt_files/resnet50/ --csv_path=./dataset_csv/label_lusc_surv.csv --model_path=/raid/Data/zhangyi/output/mil --cv_fold=5 --val_ratio=0 --model=ibmil --loss=nll_surv --weight_decay=1e-5 --lr=2e-4 --seed=2021 --title=lusc_ibmil_seed2021_k2_gated --opt=adam --lr_sche=cosine --num_workers=2 --input_dim=1024 --wandb --confounder_path=/home/zhangyi/tangwenhao/mil_new/datasets_deconf/LUSC/resnet50 --confounder_k=2



