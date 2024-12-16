# Train Shenzhen dataset
# shell命令中只能用#注释

weight_save_dir=(
    "./weights_sz_v1_add"
    "./weights_sz_v2_add"
    "./weights_sz_v3_add"
)
models=(
    "unet"
    "resunet"
    "deeplabv3+"
    "linknet"
    "dlink34_1d"
    "dlink34"
)
for w in "${weight_save_dir[@]}"; do
    for m in "${models[@]}"; do
        if [ "${m}" == "resunet" ]; then
            n=2
        else
            n=1
        fi
        # # vanilla, GPS
        # ts -G "${n}" python train.py \
        #     --model "${m}" \
        #     --weight_save_dir "${w}" \
        #     --dataset_name sz \
        #     --sat_dir '' \
        #     --mask_dir datasets/dataset_sz_grid/train_val/mask \
        #     --test_sat_dir '' \
        #     --test_mask_dir datasets/dataset_sz_grid/test/mask \
        #     --gps_dir datasets/dataset_sz_grid/GPS/taxi \
        #     \
        #     --gps_type data \
        #     --gps_render_type count \
        #     --count_render_type direct
        # # vanilla, Satellite
        # ts -G "${n}" python train.py \
        #     --model "${m}" \
        #     --weight_save_dir "${w}" \
        #     --dataset_name sz \
        #     --sat_dir datasets/dataset_sz_grid/train_val/image \
        #     --mask_dir datasets/dataset_sz_grid/train_val/mask \
        #     --test_sat_dir datasets/dataset_sz_grid/test/image_test \
        #     --test_mask_dir datasets/dataset_sz_grid/test/mask \
        #     --gps_dir '' \
        #     \
        #     --gps_type '' \
        #     --gps_render_type '' \
        #     --count_render_type ''
        # vanilla, GPS+Satellite
        ts -G "${n}" python train.py \
            --model "${m}" \
            --weight_save_dir "${w}" \
            --dataset_name sz \
            --sat_dir datasets/dataset_sz_grid/train_val/image \
            --mask_dir datasets/dataset_sz_grid/train_val/mask \
            --test_sat_dir datasets/dataset_sz_grid/test/image_test \
            --test_mask_dir datasets/dataset_sz_grid/test/mask \
            --gps_dir datasets/dataset_sz_grid/GPS/taxi \
            \
            --gps_type data \
            --gps_render_type count \
            --count_render_type direct
    done
done
