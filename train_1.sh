CONFIG=depth_occ_lovasz_non_visible_ignore
CONFIG_PATH=MTL_baseline
# ./tools/dist_train.sh projects/configs/$CONFIG_PATH/$CONFIG.py 4 --work-dir ./work_dirs/$CONFIG_PATH/$CONFIG
PORT=14725 CUDA_VISIBLE_DEVICES=0 tools/dist_test_plus.sh projects/configs/$CONFIG_PATH/$CONFIG.py ./work_dirs/$CONFIG_PATH/$CONFIG/epoch_24_ema.pth 1 --eval bbox  >> ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt
python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt 11


# CONFIG=depth_O_S
# CONFIG_PATH=MTL_baseline
# ./tools/dist_train.sh projects/configs/$CONFIG_PATH/$CONFIG.py 4 --work-dir ./work_dirs/$CONFIG_PATH/$CONFIG
# PORT=14723 CUDA_VISIBLE_DEVICES=0 tools/dist_test_plus.sh projects/configs/$CONFIG_PATH/$CONFIG.py ./work_dirs/$CONFIG_PATH/$CONFIG/epoch_24_ema.pth 1 --eval bbox  >> ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt
# python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt 11



# CONFIG=depth_D_O_S3
# CONFIG_PATH=MTL_baseline
# ./tools/dist_train.sh projects/configs/$CONFIG_PATH/$CONFIG.py 4 --work-dir ./work_dirs/$CONFIG_PATH/$CONFIG
# PORT=14723 CUDA_VISIBLE_DEVICES=0 tools/dist_test_plus.sh projects/configs/$CONFIG_PATH/$CONFIG.py ./work_dirs/$CONFIG_PATH/$CONFIG/epoch_24_ema.pth 1 --eval bbox  >> ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt
# python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt 11
