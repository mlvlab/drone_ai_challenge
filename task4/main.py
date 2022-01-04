import os
import argparse
import json
import time
#task3
import task3.lib 
import task3.lib.predict as pred

#task2
from task2.final_1st.lib.Task2 import *

#task4
from task4.main_inference import *

#task1
from task1.run import *


def make_final_json(task1_answer,task2_answer, task3_answer,task4_answer):
    final_json = dict()
    final_json["task1_answer"] = task1_answer
    final_json["task2_answer"] = task2_answer
    final_json["task3_answer"] = task3_answer
    final_json["task4_answer"] = task4_answer

    with open(json_path, 'w', encoding='utf-8') as make_file:
        json.dump(final_json, make_file,ensure_ascii=False,indent=3 )

def save_task3_answer(set_num, pred_data):
    
    data["task3_answer"].append({f"{set_num}": pred_data})

    return data
    


def main(args):
    start = time.time()
    if os.path.exists(args.json_path) == True:
        os.remove(args.json_path)

    data ={}
    data["task3_answer"] = []
    
    #n_set = 5 if args.set_num == "all_sets" else 1
    #print(args.set_num)
    # task_2 - 5 set 에 대해서 이미 다 구현
    # data_path 나중에 대회 데이터셋 경로로 바꾸기 
    data_path = '/home/shinyeong/final_dataset/'
    #~~~~~~~~~ task_2 ~~~~~~~~~~~ 
    
    start_task2 = time.time()
    
    task2_answer = task2_inference(data_path,5)
    print("TASK2 : ",time.time()-start_task2)
    print(task2_answer)
    
    '''
    #~~~~~~~~~ task_4 ~~~~~~~~~~~ 
    #data_path = '/home/shinyeong/final_dataset/'
    start_task4 = time.time()
    task4_answer = task4_main(data_path)
    print("TASK4 : ",time.time() - start_task4)
    print(task4_answer)
    

    ##############dataset 경로 절대경로로 넣어야함!!!!!################
    task1_answer = [{
                        "set_1": [],
                        "set_2": [],
                        "set_3": [],
                        "set_4": [],
                        "set_5": []
                    }]
    for i in range(5): #n_set : number of set
        args.set_num = f"set_0" + str(i+1)
        #print("count : ", args.set_num)
        set_num = f"set_0" + str(i+1)
        #~~~~~~~~~ task_1 ~~~~~~~~~~~
        
        task1_video_path = data_path + set_num #final_dataset/set_01
        task1_img_path = data_path + set_num
        task1_frame_skip = 15
        task1_main(task1_video_path, task1_img_path, task1_frame_skip, task1_answer)
        
        
        #~~~~~~~~~ task_3 ~~~~~~~~~~~
        
        t3_data = []
        t3 = pred.func_task3(args)
        t3_res_pred_move, t3_res_pred_stay, t3_res_pred_total = t3.run()
        t3_data.append(t3_res_pred_move)
        t3_data.append(t3_res_pred_stay)
        t3_data.append(t3_res_pred_total)
        task3_answer = save_task3_answer(set_num,t3_data)
        
        #~~~~~~~~~ task_4 ~~~~~~~~~~~        

        
    

    make_final_json(task1_answer,task2_answer, task3_answer,task4_answer)
    print("TOTAL INFERENCE TIME : ", time.time()-start)
    '''
    
if __name__ == '__main__':
    p=argparse.ArgumentParser()
    # path
    # p.add_argument("--dataset_dir", type=str, default="/home/agc2021/dataset") # /set_01, /set_02, /set_03, /set_04, /set_05
    # p.add_argument("--root_dir", type=str, default="/home/[Team_ID]")
    # p.add_argument("--temporary_dir", type=str, default="/home/agc2021/temp")
    ###
    json_path = "answersheet_3_00_Rony2.json"
    p.add_argument("--dataset_dir", type=str, default="/home/shinyeong/final_dataset") # /set_01, /set_02, /set_03, /set_04, /set_05
    p.add_argument("--root_dir", type=str, default="./")
    p.add_argument("--temporary_dir", type=str, default="../output3")
    
    ###
    p.add_argument("--json_path", type=str, default="answersheet_3_00_Rony2.json")
    p.add_argument("--task_num", type=str, default="task3_answer")
    p.add_argument("--set_num", type=str, default="all_set") 
    p.add_argument("--device", default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    p.add_argument("--test", type = int, default = '3', help = 'number of video,3')
    p.add_argument("--release_mode", type=bool, default = True)
    
    args = p.parse_args()

    main(args)

    

