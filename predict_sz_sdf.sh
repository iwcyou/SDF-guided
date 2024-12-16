#Predict Shenzhen dataset

# sdf, filtered, gaussian, dlink34、gps、sat, prediction
python train.py \
    --model dlink34_1d \
    --dataset_name sz \
    --sat_dir ../datasets/dataset_sz_grid/train_val/image \
    --mask_dir ../datasets/dataset_sz_grid/train_val/mask_sdf_T \
    --test_sat_dir ../datasets/dataset_sz_grid/test/image_test \
    --test_mask_dir ../datasets/dataset_sz_grid/test/mask_sdf_T \
    --gps_dir ../datasets/dataset_sz_grid/GPS/taxi \
    \
    --mask_type sdf \
    --gps_type data \
    --gps_render_type count \
    --quantity_render_type direct \
    --wandb_group "predict" \
    --wandb_notes "predict Shenzhen dataset" \
	--eval predict \
	--weight_load_path /home/fk/python_code/traj/weights_sz_sdf_seed_0/epoch88_val0.5833_test0.9394_0.7014_0.7137_0.5519_0.6957.pth
