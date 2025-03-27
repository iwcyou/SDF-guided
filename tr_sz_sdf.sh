# Train Shenzhen dataset sdf mask
# shell命令中只能用#注释

weight_save_dir=(
    "./weights_sz_sdf_diff_source"
    # "./weights_sz_latent_v2"
    # "./weights_sz_v3"
)
models=(
    # "unet"
    # "resunet"
    # "deeplabv3+"
    # "linknet"
    # "dlink34"
    "dlink34_1d"
)
delta=(
    # 1
    10
    # 20
    # 40
    # 60
    # 80
)
loss_weight=(
    # 0
    # 0.1
    1
    # # 10
    # 100
    # # 1000
    # # 10000
)
for w in "${weight_save_dir[@]}"; do
    for m in "${models[@]}"; do
        if [ "${m}" == "resunet" ]; then
            n=2
        else
            n=1
        fi
        for delta in "${delta[@]}"; do
            for weight in "${loss_weight[@]}"; do
                # # sdf mask, ltqs, filtered, GPS+Satellite, gaussian
                # ts -G "${n}" python train.py \
                #     --model "${m}" \
                #     --weight_save_dir "${w}" \
                #     --dataset_name sz \
                #     --sat_dir ../datasets/dataset_sz_grid/train_val/image \
                #     --mask_dir ../datasets/dataset_sz_grid/train_val/mask_sdf_T \
                #     --test_sat_dir ../datasets/dataset_sz_grid/test/image_test \
                #     --test_mask_dir ../datasets/dataset_sz_grid/test/mask_sdf_T \
                #     --gps_dir ../datasets/dataset_sz_grid/GPS/taxi_filtered_time_quantity_speed_gaussian_patch \
                #     \
                #     --mask_type sdf \
                #     --gps_type image \
                #     --gps_render_type filtered_gaussian_ltqs \
                #     --quantity_render_type log \
                #     --epochs 200 \
                #     --wandb_group sz_exp\
                #     --wandb_notes "This sz dataset run used mese+dice+bce loss." \
                #     --delta "${delta}" \
                #     --loss mse_dice_bce_loss \
                #     --loss_weight "${weight}"
                # # vanilla, GPS+Satellite, direct
                # ts -G "${n}" python train.py \
                #     --model "${m}" \
                #     --weight_save_dir "${w}" \
                #     --dataset_name sz \
                #     --sat_dir ../datasets/dataset_sz_grid/train_val/image \
                #     --mask_dir ../datasets/dataset_sz_grid/train_val/mask \
                #     --test_sat_dir ../datasets/dataset_sz_grid/test/image_test \
                #     --test_mask_dir ../datasets/dataset_sz_grid/test/mask \
                #     --gps_dir ../datasets/dataset_sz_grid/GPS/taxi \
                #     \
                #     --mask_type png \
                #     --gps_type data \
                #     --gps_render_type count \
                #     --quantity_render_type direct \
                #     --epochs 100 \
                #     --wandb_group sz_overall\
                #     --wandb_notes "sz overall exp." \
                #     --delta "${delta}" \
                #     --loss dice_bce_loss \
                #     --loss_weight "${weight}"
                # sdf mask, GPS+Satellite, direct
                ts -G "${n}" python train.py \
                    --model "${m}" \
                    --weight_save_dir "${w}" \
                    --dataset_name sz \
                    --sat_dir ../datasets/dataset_sz_grid/train_val/image \
                    --mask_dir ../datasets/dataset_sz_grid/train_val/mask_sdf_T \
                    --test_sat_dir ../datasets/dataset_sz_grid/test/image_test \
                    --test_mask_dir ../datasets/dataset_sz_grid/test/mask_sdf_T \
                    --gps_dir /home/fk/python_code/datasets/dataset_sz_grid/GPS/taxi \
                    \
                    --mask_type sdf \
                    --gps_type data \
                    --gps_render_type count \
                    --quantity_render_type direct \
                    --epochs 200 \
                    --wandb_group sz_overall\
                    --wandb_notes "sz overall exp." \
                    --delta "${delta}" \
                    --loss mse_dice_bce_loss \
                    --loss_weight "${weight}"
                # # SDF, GPS
                # ts -G "${n}" python train.py \
                #     --model "${m}" \
                #     --weight_save_dir "${w}" \
                #     --dataset_name sz \
                #     --sat_dir '' \
                #     --mask_dir ../datasets/dataset_sz_grid/train_val/mask_sdf_T \
                #     --test_sat_dir '' \
                #     --test_mask_dir ../datasets/dataset_sz_grid/test/mask_sdf_T \
                #     --gps_dir /home/fk/python_code/datasets/dataset_sz_grid/GPS/taxi \
                #     \
                #     --mask_type sdf \
                #     --gps_type data \
                #     --gps_render_type count \
                #     --quantity_render_type direct \
                #     --epochs 100 \
                #     --wandb_group sz_diff_source\
                #     --wandb_notes "sz different resource exp." \
                #     --delta "${delta}" \
                #     --loss mse_dice_bce_loss \
                #     --loss_weight "${weight}"
                # # SDF, Satellite
                # ts -G "${n}" python train.py \
                #     --model "${m}" \
                #     --weight_save_dir "${w}" \
                #     --dataset_name sz \
                #     --sat_dir ../datasets/dataset_sz_grid/train_val/image \
                #     --mask_dir ../datasets/dataset_sz_grid/train_val/mask_sdf_T \
                #     --test_sat_dir ../datasets/dataset_sz_grid/test/image_test \
                #     --test_mask_dir ../datasets/dataset_sz_grid/test/mask_sdf_T \
                #     --gps_dir '' \
                #     \
                #     --mask_type sdf \
                #     --gps_type '' \
                #     --gps_render_type '' \
                #     --quantity_render_type '' \
                #     --epochs 100 \
                #     --wandb_group sz_diff_source\
                #     --wandb_notes "sz different resource exp." \
                #     --delta "${delta}" \
                #     --loss mse_dice_bce_loss \
                #     --loss_weight "${weight}"
            done
        done
    done
done
