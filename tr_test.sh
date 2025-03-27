# Train Shenzhen dataset sdf mask
# shell命令中只能用#注释

weight_save_dir=(
    "./weights_sz_sdf_lots_test_seed"
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
random_seed=(
    # 0
    # 42
    1017
    # 8888
)
for w in "${weight_save_dir[@]}"; do
    for m in "${models[@]}"; do
        for seed in "${random_seed[@]}"; do
            if [ "${m}" == "resunet" ]; then
                n=2
            else
                n=1
            fi
            for delta in "${delta[@]}"; do
                for weight in "${loss_weight[@]}"; do
                    # sdf mask, GPS+Satellite, direct
                    ts -G "${n}" python train.py \
                        --model "${m}" \
                        --random_seed "${seed}" \
                        --weight_save_dir "${w}_${seed}" \
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
                        --wandb_group sz_sdf_test\
                        --wandb_notes "sz sdf test exp; try directly reading seg mask." \
                        --delta "${delta}" \
                        --loss mse_dice_bce_loss \
                        --loss_weight "${weight}"
                    # sdf mask, GPS+Satellite, direct, mse loss
                    ts -G "${n}" python train.py \
                        --model "${m}" \
                        --random_seed "${seed}" \
                        --weight_save_dir "${w}_${seed}" \
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
                        --wandb_group sz_sdf_test\
                        --wandb_notes "sz sdf test exp, try directly reading seg mask; mse loss." \
                        --delta "${delta}" \
                        --loss mse_loss \
                        --loss_weight "${weight}"
                done
            done
        done
    done
done
