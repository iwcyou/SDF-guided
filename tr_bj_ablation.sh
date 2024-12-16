# Train Beijing dataset
# shell命令中只能用#注释

weight_save_dir=(
    "./weights_bj_baseline"
    # "./weights_bj_sdf_v2"
    # "./weights_bj_sdf_v4"
)
models=(
    # "unet"
    # "resunet"
    # "deeplabv3+"
    # "linknet"
    # "dlink34_1d"
    "dlink34"
)
delta=(
    # 0
    # 10
    20
    # 40
    # 60
    # 80
)
for w in "${weight_save_dir[@]}"; do
    for m in "${models[@]}"; do
        if [ "${m}" == "resunet" ]; then
            n=2
        else
            n=1
        fi
        for delta in "${delta[@]}"; do
            # # ltqs, GPS
            # ts -G "${n}" python train.py \
            #     --model "${m}" \
            #     --weight_save_dir "${w}" \
            #     --dataset_name sz \
            #     --sat_dir '' \
            #     --mask_dir ../datasets/dataset_bj_time/train_val/mask \
            #     --test_sat_dir '' \
            #     --test_mask_dir ../datasets/dataset_bj_time/test/mask \
            #     --gps_dir ../datasets/dataset_bj_time/GPS/taxi_gaussian_ltqs_patch \
            #     \
            #     --gps_type image \
            #     --gps_render_type gaussian_ltqs \
            #     --count_render_type log
            # vanilla, GPS+Satellite, direct
            ts -G "${n}" python train.py \
                --model "${m}" \
                --weight_save_dir "${w}" \
                --dataset_name bj \
                --sat_dir ../datasets/dataset_bj_time/train_val/image \
                --mask_dir ../datasets/dataset_bj_time/train_val/mask \
                --test_sat_dir ../datasets/dataset_bj_time/test/image_test \
                --test_mask_dir ../datasets/dataset_bj_time/test/mask \
                --gps_dir ../datasets/dataset_bj_time/GPS/patch \
                \
                --mask_type png \
                --gps_type data \
                --gps_render_type count \
                --quantity_render_type direct \
                --epochs 100
            # ltqs, GPS+Satellite, gaussian
            ts -G "${n}" python train.py \
                --model "${m}" \
                --weight_save_dir "${w}" \
                --dataset_name bj \
                --sat_dir ../datasets/dataset_bj_time/train_val/image \
                --mask_dir ../datasets/dataset_bj_time/train_val/mask \
                --test_sat_dir ../datasets/dataset_bj_time/test/image_test \
                --test_mask_dir ../datasets/dataset_bj_time/test/mask \
                --gps_dir ../datasets/dataset_bj_time/GPS/gaussian_patch \
                \
                --mask_type png \
                --gps_type image \
                --gps_render_type gaussian_ltqs \
                --quantity_render_type log \
                --epochs 100
            # filtered, ltqs, GPS+Satellite, gaussian
            ts -G "${n}" python train.py \
                --model "${m}" \
                --weight_save_dir "${w}" \
                --dataset_name bj \
                --sat_dir ../datasets/dataset_bj_time/train_val/image \
                --mask_dir ../datasets/dataset_bj_time/train_val/mask \
                --test_sat_dir ../datasets/dataset_bj_time/test/image_test \
                --test_mask_dir ../datasets/dataset_bj_time/test/mask \
                --gps_dir ../datasets/dataset_bj_time/GPS/filtered_time_quantity_speed_gaussian_patch \
                \
                --mask_type png \
                --gps_type image \
                --gps_render_type filtered_gaussian_ltqs \
                --quantity_render_type log \
                --epochs 100
            # # GPS+Satellite, sdf mask
            # ts -G "${n}" python train.py \
            #     --model "${m}" \
            #     --weight_save_dir "${w}" \
            #     --dataset_name bj \
            #     --sat_dir ../datasets/dataset_bj_time/train_val/image \
            #     --mask_dir ../datasets/dataset_bj_time/train_val/mask_sdf_T \
            #     --test_sat_dir ../datasets/dataset_bj_time/test/image_test \
            #     --test_mask_dir ../datasets/dataset_bj_time/test/mask_sdf_T \
            #     --gps_dir ../datasets/dataset_bj_time/GPS/patch \
            #     \
            #     --mask_type sdf \
            #     --gps_type data \
            #     --gps_render_type count \
            #     --quantity_render_type direct \
            #     --epochs 60 \
            #     --delta "${delta}"
            # # filtered, ltqs, GPS+Satellite, gaussian
            # ts -G "${n}" python train.py \
            #     --model "${m}" \
            #     --weight_save_dir "${w}" \
            #     --dataset_name bj \
            #     --sat_dir ../datasets/dataset_bj_time/train_val/image \
            #     --mask_dir ../datasets/dataset_bj_time/train_val/mask_sdf_T \
            #     --test_sat_dir ../datasets/dataset_bj_time/test/image_test \
            #     --test_mask_dir ../datasets/dataset_bj_time/test/mask_sdf_T \
            #     --gps_dir ../datasets/dataset_bj_time/GPS/filtered_time_quantity_speed_gaussian_patch \
            #     \
            #     --mask_type sdf \
            #     --gps_type image \
            #     --gps_render_type filtered_gaussian_ltqs \
            #     --quantity_render_type log \
            #     --epochs 60 \
            #     --delta "${delta}"
        done
    done
done
