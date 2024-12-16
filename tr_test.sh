# 测试用脚本

weight_save_dir=(
    "./weights_sz_v1_add"
    # "./weights_sz_v2_add"
    # "./weights_sz_v3_add"
)
models=(
    # "unet"
    # "resunet"
    # "deeplabv3+"
    # "linknet"
    # "dlink34"
    "dlink34_1d"
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
    done
done
