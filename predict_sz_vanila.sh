#预测Shenzhen数据集


weight_save_dir=(
    "./weights_sz_v1"
    "./weights_sz_v2"
    "./weights_sz_v3"
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
        # vanilla, GPS
        ts -G 1 python train.py \
            --model "${m}" \
            --weight_save_dir "${w}" \
            --dataset_name sz \
            --sat_dir '' \
            --mask_dir datasets/dataset_sz_grid/train_val/mask \
            --test_sat_dir '' \
            --test_mask_dir datasets/dataset_sz_grid/test/mask \
            --gps_dir datasets/dataset_sz_grid/GPS/taxi \
            \
            --gps_type data \
            --gps_render_type count \
            --count_render_type direct \
            --eval predict
        # vanilla, Satellite
        ts -G 1 python train.py \
            --model "${m}" \
            --weight_save_dir "${w}" \
            --dataset_name sz \
            --sat_dir datasets/dataset_sz_grid/train_val/image \
            --mask_dir datasets/dataset_sz_grid/train_val/mask \
            --test_sat_dir datasets/dataset_sz_grid/test/image_test \
            --test_mask_dir datasets/dataset_sz_grid/test/mask \
            --gps_dir '' \
            \
            --gps_type '' \
            --gps_render_type '' \
            --count_render_type '' \
            --eval predict
        # vanilla, GPS+Satellite
        ts -G 1 python train.py \
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
            --count_render_type direct \
            --eval predict
    done
done



# #dlink34、gps、sat、ltqs预测图像,高斯渲染
# python train.py \
#     --model unet \
#     --dataset_name sz \
#     --sat_dir datasets/dataset_sz_grid/train_val/image \
#     --mask_dir datasets/dataset_sz_grid/train_val/mask \
#     --test_sat_dir datasets/dataset_sz_grid/test/image_test \
#     --test_mask_dir datasets/dataset_sz_grid/test/mask \
#     --gps_dir datasets/dataset_sz_grid/GPS/taxi_gaussian_ltqs_patch \
#     \
#     --gps_type image \
#     --gps_render_type gaussian_ltqs \
#     --count_render_type log \
# 	--eval predict \
# 	--weight_load_path ./weights_sz_v1/unet_sat_gpsimage_gaussian_ltqs_log__/epoch13_val0.5207_test0.9140_0.6166_0.5969_0.4292_0.5971_0.0211.pth
