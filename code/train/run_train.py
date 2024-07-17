""" Generate commands for meta-train phase. """
import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
def run_exp():    
    the_command = (
        'python code/train/main.py' 
        + ' --dataset_dir=' + 'data/places2'
        
    )

    os.system(the_command + ' --phase=train') #os.system：終端機執行其中指令，--dataset_dir、--phase都是在main.py自定義的參數

run_exp()
