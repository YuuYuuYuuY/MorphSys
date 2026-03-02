# bash debug_models.sh sumpool mlp 512 1 1 lstm out weighted && /usr/bin/shutdown
readout=$1
projector=$2
batch_size=$3
fixed_seed=$4
interbranch=$5
path=$6
direction=$7
fuse=$8


pool_models=(
    "gin"
    "gat"
    "agnn"
    "appnp"
    "dgn"
    "dotgat"
    "gatedGraph"
    "gcn2 "
    "pna"
    "sgc"
    "tag"
    "twirls"
    "sage"
)

if [ "$fixed_seed" -eq 1 ]; then
    if [ "$interbranch" -eq 1 ]; then
        for pool_model in "${pool_models[@]}";
        do
        python3 train_contrastive_all.py \
            --epochs 100 \
            --fixed_seed $fixed_seed \
            --exp_name fixed_seed_${direction}_with_batch_size_${batch_size}_${path}_both_${fuse} \
            --batch_size 512 \
            --bn \
            --projector_bn \
            --aug_scale_coords \
            --aug_jitter_coords \
            --aug_rotate \
            --aug_shift_coords \
            --aug_flip \
            --aug_mask_feats \
            --aug_jitter_length \
            --save_freq 100 \
            --knn \
            --eval_act \
            --eval_jm \
            --encoder $pool_model \
            --readout $readout \
            --projector $projector \
            --interbranch $interbranch \
            --path $path \
            --direction $direction \
            --fuse $fuse
        done
    else
        for pool_model in "${pool_models[@]}";
        do
        python3 train_contrastive_all.py \
            --epochs 100 \
            --fixed_seed $fixed_seed \
            --exp_name fixed_seed_with_batch_size_${batch_size}_gnn \
            --batch_size 512 \
            --bn \
            --projector_bn \
            --aug_scale_coords \
            --aug_jitter_coords \
            --aug_rotate \
            --aug_shift_coords \
            --aug_flip \
            --aug_mask_feats \
            --aug_jitter_length \
            --save_freq 100 \
            --knn \
            --eval_act \
            --eval_jm \
            --encoder $pool_model \
            --readout $readout \
            --projector $projector \
            --interbranch $interbranch \
            --path $path \
            --direction $direction
        done
    fi
    echo "yes"
else
    echo "no"
fi



