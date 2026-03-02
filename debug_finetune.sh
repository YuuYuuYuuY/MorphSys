# bash debug_finetune.sh sumpool mlp 512 1 1 add out weighted && /usr/bin/shutdown
readout=$1
projector=$2
batch_size=$3
fixed_seed=$4
interbranch=$5
path=$6
direction=$7
fuse=$8

pool_models=(
    # "gin"
    # "gat"
    # "agnn"
    # "appnp"
    # "dgn"
    # "dotgat"
    "gatedGraph"
    # "gcn2 "
    # "pna"
    # "sgc"
    # "tag"
    # "twirls"
    # "sage"
)
if [ "$interbranch" -eq 1 ]; then
    for pool_model in "${pool_models[@]}";
    do
    python3 lincls_or_finetune.py \
        --epochs 200 \
        --exp_name finetune_${direction}_with_batch_size_${batch_size}_${path}_both_${fuse} \
        --batch_size 512 \
        --mode finetune \
        --bn \
        --dataset ACT \
        --aug_scale_coords \
        --aug_jitter_coords \
        --aug_rotate \
        --aug_shift_coords \
        --aug_flip \
        --aug_mask_feats \
        --aug_jitter_length \
        --save_freq 200 \
        --encoder $pool_model \
        --readout $readout \
        --projector $projector \
        --interbranch $interbranch \
        --path $path \
        --direction $direction \
        --fuse $fuse \
        --pretrained work_dir/fixed_seed_${direction}_with_batch_size_${batch_size}_${path}_both_${fuse}/checkpoints/${pool_model}_${readout}_${projector}_${path}_epoch_100.pth
    done
else
    echo "not implemented"
fi