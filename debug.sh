# bash debug.sh gin sumpool mlp 512 1 1 1 lstm out
gnn=$1
readout=$2
projector=$3
batch_size=$4
loop_times=$5
fixed_seed=$6
interbranch=$7
path=$8
direction=$9

echo "Encoder: $gnn"
echo "Readout: $readout"
echo "Projector: $projector"
echo "Batch_size: $batch_size"
echo "Loop_times: $loop_times"
echo "fixed_seed: $fixed_seed"
echo "interbranch: $interbranch"
echo "path: $path"
echo "direction: $direction"

if [ "$fixed_seed" -eq 1 ]; then
    for((i=0; i<loop_times; i++))
    do
        if [ "$interbranch" -eq 1 ]; then
            python3 train_contrastive_all.py \
            --epochs 100 \
            --fixed_seed $fixed_seed \
            --exp_name fixed_seed_interbranch_with_batch_size_$batch_size \
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
            --encoder $gnn \
            --readout $readout \
            --projector $projector \
            --interbranch $interbranch \
            --path $path \
            --direction $direction
        else
            python3 train_contrastive_all.py \
            --epochs 100 \
            --fixed_seed $fixed_seed \
            --exp_name fixed_seed_with_batch_size_$batch_size \
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
            --encoder $gnn \
            --readout $readout \
            --projector $projector \
            --interbranch $interbranch \
            --path $path \
            --direction $direction
        fi
    done
else
    for((i=0; i<loop_times; i++))
    do
        if [ "$interbranch" -eq 1 ]; then
            python3 train_contrastive_all.py \
            --epochs 100 \
            --fixed_seed $fixed_seed \
            --exp_name random_seed_interbranch_with_batch_size_$batch_size \
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
            --encoder $gnn \
            --readout $readout \
            --projector $projector \
            --interbranch $interbranch \
            --path $path \
            --direction $direction
        else
            python3 train_contrastive_all.py \
            --epochs 100 \
            --fixed_seed $fixed_seed \
            --exp_name random_seed_with_batch_size_$batch_size \
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
            --encoder $gnn \
            --readout $readout \
            --projector $projector \
            --interbranch $interbranch \
            --path $path \
            --direction $direction
        fi
    
    done
fi
