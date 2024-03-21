

CONFIG=depth_occ
CONFIG_PATH=MTL_baseline
# ./tools/dist_train.sh projects/configs/$CONFIG_PATH/$CONFIG.py 4 --work-dir ./work_dirs/$CONFIG_PATH/$CONFIG
# PORT=14723 CUDA_VISIBLE_DEVICES=0 tools/dist_test_plus.sh projects/configs/$CONFIG_PATH/$CONFIG.py ./work_dirs/$CONFIG_PATH/$CONFIG/epoch_24_ema.pth 1 --eval bbox  >> ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt
# PORT=12398 CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh ./projects/configs/$CONFIG_PATH/$CONFIG.py ./work_dirs/$CONFIG_PATH/$CONFIG/epoch_2.pth 1 --eval mAp #>> ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt
# PORT=12398 CUDA_VISIBLE_DEVICES=0 ./tools/dist_test.sh ./projects/configs/$CONFIG_PATH/$CONFIG.py ./work_dirs/$CONFIG_PATH/$CONFIG/epoch_24_ema.pth 1 --eval mAp >> ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt
python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt 11
# 