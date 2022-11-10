# pre-training the source model
CUDA_VISIBLE_DEVICES=* GPUS_PER_NODE=n ./tools/run_dist_launch.sh n ./configs/r50_deformable_detr.sh --output_dir exps/source_model --dataset_file city2foggy_source

# training the proposed method
CUDA_VISIBLE_DEVICES=* GPUS_PER_NODE=n ./tools/run_dist_launch.sh n ./configs/DA_r50_deformable_detr.sh --output_dir exps/ours --transform make_da_transforms --dataset_file city2foggy --checkpoint exps/source_model/checkpoint.pth


