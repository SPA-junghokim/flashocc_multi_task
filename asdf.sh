
CONFIG=dy_MatrixVT_2Dbone_3Dreshape_Maskhead_NoLayer_Aux
CONFIG_PATH=DY_baseline
python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result12.txt dylee


CONFIG=dy_BEVpool_2Dbone_3Dreshape_Maskhead_NoLayer_Aux_Depthconv4
CONFIG_PATH=DY_baseline
python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result12.txt dylee


# CONFIG=dy_MatrixVT_2Dbone_3Dreshape_Maskhead_NoLayer_Aux
# CONFIG_PATH=DY_baseline
# python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt dylee


# CONFIG=dy_MatrixVT_2Dbone_3Dreshape_Maskhead_NoLayer_Aux
# CONFIG_PATH=DY_baseline
# python send_result_mail.py ./work_dirs/$CONFIG_PATH/$CONFIG/result.txt dylee

