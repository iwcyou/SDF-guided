#Predict Beijing dataset

# #dlink34、gps、sat、ltqs预测图像,高斯渲染
# python train.py \
#     --model dlink34 \
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
# 	--eval predict \
# 	--weight_load_path /home/fk/python_code/traj/weights_bj_baseline/dlink34_sat_gpsdata_png_count_direct_20__/epoch25_val0.5835_test0.9401_0.8174_0.7263_0.6225_0.7661_0.0000.pth


#dlink34、gps、sat、vanilla prediction,
python train.py \
    --model dlink34 \
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
	--eval predict \
	--weight_load_path /home/fk/python_code/traj/weights_bj_baseline/dlink34_sat_gpsdata_png_count_direct_20__/epoch56_val0.6085_test0.9469_0.7705_0.7872_0.6360_0.7765_0.0000.pth
