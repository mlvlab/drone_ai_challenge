import argparse
from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import os
import glob
import numpy as np
#parser = argparse.ArgumentParser()
#parser.add_argument('--path', type=str, default='task3/dataset_path/')
#args = parser.parse_args()
def task4_main(path):
    # ##### person #####
    config_file_person = "task4/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
    checkpoint_file_person = "task4/checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
    # config_file_person = "configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py"
    # checkpoint_file_person = "checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
    # ##################
    # ##### triage #####
    config_file_triage = "task4/configs/triage_config/triage_config.py"
    checkpoint_file_triage = "task4/work_dirs/0to3_500/ver7/epoch_41.pth"
    # config_file_triage = "configs/triage_config/triage_config.py"
    # checkpoint_file_triage = "work_dirs/0to3_500/ver7/epoch_41.pth"
    # ##################
    # # build the model from a config file and a checkpoint file
    model_person = init_detector(config_file_person, checkpoint_file_person, device="cuda:0")
    model_triage = init_detector(config_file_triage, checkpoint_file_triage, device="cuda:0")
    # set(번호)_drone(번호)_triage(번호).jpg
    # set(1~5)_drone(1~3)_triage(1~3).jpg
    # im_folder = "dataset_path/set0"
    im_folder = path
    # 제출 레이블 형식 : set_num(1~5), drone_num(1~3), frame_name[사망, 긴급, 응급, 비응급]
    #import pdb; pdb.set_trace()
    person_results = []
    set_keys = ["set_1", "set_2", "set_3", "set_4", "set_5"]
    task4_answer = dict.fromkeys(set_keys)
    # #### Extract Region of Person ####
    for set_n in range(1,6):
        set_dict = dict()
        set_name = "set_"+ str(set_n)
        set_dir= im_folder + "set_0" + str(set_n) + "/"
        # print("set_dir : ",set_dir)
        for filename in glob.glob(set_dir): # filename : dataset_path/set_01/
            # print("filename: ", filename)
            drone_1, drone_2, drone_3 = dict(), dict(), dict()
            file_list = os.listdir(filename)
            file_list = sorted(file_list)
            triage_list=[]
            # set_dict_list = []
            for x in file_list:
                if 'triage' in x :
                    #file_list = os.listdir(filename) # ["set02_drone03_triage02.jpg", "set02_drone01_triage01.jpg", "set02_drone01_triage03.jpg", "set02_drone03_triage01.jpg", "set02_drone01_triage02.jpg", "set02_drone02_triage01.jpg", "set02_drone02_triage02.jpg"]
                    triage_list.append(x)
            drone_dict_list = dict()
            for file_idx in range(len(triage_list)):
                # print("file_idx: ", file_idx)
                answer_sheet = [0 for i in range(4)]
                ori_file = set_dir + triage_list[file_idx]
                # # print(file_list[i]) # set01_drone01_triage01.jpg
                # # print(set_dir + file_list[i]) # dataset_path/set01/set01_drone01_triage01.jpg
                result_person = inference_detector(model_person, ori_file) # person
                result_triage = inference_detector(model_triage, ori_file) # person
                labels = [
                    np.full(bbox.shape[0], idx, dtype=np.int32)
                    for idx, bbox in enumerate(result_triage)
                ]
                labels = np.concatenate(labels)
                bboxes_person = np.vstack(result_person)
                bboxes_triage = np.vstack(result_triage)
                # class-based NMS
                tag_is_in_person = [False for t in range(bboxes_triage.shape[0])]
                for p in range(bboxes_person.shape[0]):
                    person_pos = bboxes_person[p][:4] # each person has one or no triage tag.
                    is_exist_pos = []
                    max_score = -1
                    if bboxes_person[p][-1] < 0.5: continue
                    for q in range(bboxes_triage.shape[0]):
                        triage_pos = bboxes_triage[q][:4]
                        if(triage_pos[0] >= person_pos[0]-50 and triage_pos[1] >= person_pos[1]-50 and triage_pos[2] <= person_pos[2]+50 and triage_pos[3] <= person_pos[3]+50):
                            is_exist_pos.append(q)
                            tag_is_in_person[q] = True
                            if bboxes_triage[q][-1] > max_score:
                                max_score = bboxes_triage[q][-1]
                    for k in range(len(is_exist_pos)):
                        if bboxes_triage[is_exist_pos[k]][-1] < max_score:
                            bboxes_triage[is_exist_pos[k]][-1] = 0
                for t in range(bboxes_triage.shape[0]):
                    if tag_is_in_person[t] == False:
                        bboxes_triage[t][-1] = 0
                scores = bboxes_triage[:, -1]
                score_thr = 0.3
                inds = scores > score_thr
                labels = labels[inds]
                for i in range(len(labels)):
                    answer_sheet[labels[i]] += 1
                drone_num = int(triage_list[file_idx].split("drone")[1][:2])
                drone = "drone_" + str(drone_num)
                img_key = triage_list[file_idx].split(".jpg")[0]
                if drone_num == 1:
                    drone_1[img_key] = answer_sheet
                elif drone_num == 2:
                    drone_2[img_key] = answer_sheet
                elif drone_num == 3:
                    drone_3[img_key] = answer_sheet
            drone_1_list, drone_2_list, drone_3_list = [], [], []
            drone_1_list.append(drone_1)
            drone_2_list.append(drone_2)
            drone_3_list.append(drone_3)
            set_drone1, set_drone2, set_drone3 = dict(), dict(), dict()
            set_drone1["drone_1"] = drone_1_list
            set_drone2["drone_2"] = drone_2_list
            set_drone3["drone_3"] = drone_3_list
            set_drone_list = []
            set_drone_list.append(set_drone1)
            set_drone_list.append(set_drone2)
            set_drone_list.append(set_drone3)
            task4_answer[set_name] = set_drone_list
            # print("\n \ntask4_answer[set_name] :\n ", task4_answer[set_name])
    # print(task4_answer)
    #final_answer = dict()
    #final_answer["task4_answer"] = task4_answer
    return task4_answer
if __name__ == '__main__':
    # path = 'dataset_path/'
    task4_main(path)