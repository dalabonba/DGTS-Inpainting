""" Generate commands for meta-train phase. """
import os
import math

# 設置可見的 CUDA 設備，這裡指定使用 GPU 0, 1, 2, 3
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def run_exp():    
    # 構建執行訓練的命令
    the_command = (
        'python code/train/main.py'  # 執行 main.py 腳本
        + ' --dataset_dir=' + 'data/places2'  # 設置資料集目錄，於main.py有預設值data/places2
    )

    # 執行構建好的命令
    # os.system() 函數用於在終端執行命令
    os.system(the_command + ' --phase=train') # phase屬性對程式不影響(沒用到)

run_exp()
