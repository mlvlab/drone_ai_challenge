from task2.lib.Task2 import *

# data_path = '/home/agc2021/dataset/'
data_path = '../temp_data'

# Task 2
set_nums = 5
json_str = task2_inference(data_path, set_nums)

''' Below code is for sanity check!!'''
''' Expected print for set_nums = 5 '''
'''     Total s, d, i, er, correct: 15 15 0 0.46 50    '''
# from lib.marg_utils import *
# import json
# json_dict = json.loads(json_str)
# answer_list = json_to_answer(json_dict, set_nums)
# evaluation(answer_list, set_nums)