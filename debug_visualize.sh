# bash debug_visualize.sh gin sumpool mlp 512 1 1 1 lstm out
# visualize_single(){
#         gnn=$1
#         readout=$2
#         projector=$3
#         batch_size=$4
#         fixed_seed=$5
#         interbranch=$6
#         path=$7
#         direction=$8

#         echo "Encoder: $gnn"
#         echo "Readout: $readout"
#         echo "Projector: $projector"
#         echo "Batch_size: $batch_size"
#         echo "fixed_seed: $fixed_seed"
#         echo "interbranch: $interbranch"
#         echo "path: $path"
#         echo "direction: $direction"
#         # train
#         python3 train_contrastive_all.py \
#                 --epochs 100 \
#                 --fixed_seed $fixed_seed \
#                 --exp_name visualize_interbranch_$batch_size \
#                 --batch_size 512 \
#                 --bn \
#                 --projector_bn \
#                 --aug_scale_coords \
#                 --aug_jitter_coords \
#                 --aug_rotate \
#                 --aug_shift_coords \
#                 --aug_flip \
#                 --aug_mask_feats \
#                 --aug_jitter_length \
#                 --save_freq 100 \
#                 --knn \
#                 --eval_act \
#                 --eval_jm \
#                 --encoder $gnn \
#                 --readout $readout \
#                 --projector $projector \
#                 --interbranch $interbranch \
#                 --path $path \
#                 --direction $direction \
#                 # --resume work_dir/test_visualize/checkpoints/twirls_maxpool_mlp_epoch_100.pth
# }

# visualize_single gatedGraph sumpool mlp 512 1 1 add out
# visualize_single twirls maxpool mlp 512 1 1 add out
# visualize_single gatedGraph sumpool mlp 512 1 1 lstm out

python3 train_contrastive_all.py \
        --epochs 1 \
        --exp_name test_visualize \
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
        --save_freq 1 \
        --knn \
        --eval_act \
        --eval_jm \
        --encoder gatedGraph \
        --readout sumpool \
        --projector mlp \
        --interbranch 1 \
        --path lstm \
        --direction out \
        --resume work_dir/visualize_interbranch_512/checkpoints/gatedGraph_sumpool_mlp_epoch_100.pth