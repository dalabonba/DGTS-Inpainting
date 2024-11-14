""" Generate commands for meta-train phase. """
import os
import math

# 設置 CUDA 使用的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 定義主函數，用於運行實驗
def run_exp():
    # 生成主要命令字串
    the_command = (
        'python code/test/main.py' # 執行 Python 主程式的路徑和名稱
        # + ' --dataset_dir=' + 'data/teeth_seem_inlay' # ！！！資料集目錄！！！
        
    )

    os.system(the_command + ' --phase=test') # 通過系統調用來執行命令，並設定階段為測試階段

# 執行主函數
run_exp()
