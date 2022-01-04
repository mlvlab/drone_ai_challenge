import torch
import json
import numpy as np

def convert_intseclist_to_strseclist(int_sec_result):
    str_sec_list = []
    for int_sec in int_sec_result:
        if int_sec == None:
            strsec = 'NONE'
            str_sec_list.append(strsec)
        else:
            min_str = str(int(int_sec // 60))
            sec_str = str(int(int_sec % 60))
            strsec = min_str.zfill(2) + ':' + sec_str.zfill(2)
            str_sec_list.append(strsec)

    return str_sec_list

def convert_strseclist_to_intseclist(str_sec_result):
    int_sec_list = []
    for str_sec in str_sec_result:
        if str_sec == 'NONE':
            intsec = None
            int_sec_list.append(intsec)
        else:
#             min_str = str(int(int_sec // 60))
#             sec_str = str(int(int_sec % 60))
#             strsec = min_str.zfill(2) + ':' + sec_str.zfill(2)
#             str_sec_list.append(strsec)
            int_sec_min = int(str_sec.split(':')[0])
            int_sec_sec = int(str_sec.split(':')[1])
            int_sec = int_sec_min * 60 + int_sec_sec
            int_sec_list.append(int_sec)
    return int_sec_list

def organize_result_data(result_data):
    set = 'set_'+str(result_data['set_num'])
    structured_data = {set:[]}

    for drone_num, inference_list in enumerate(result_data['result']):
        drone_name = 'drone_'+str(drone_num+1)
        drone_result_dic = {drone_name:None}
        class_result_dic = {}
        for class_index, class_result in enumerate(inference_list):
            class_key_list = ['M', 'W', 'C']
            class_key = class_key_list[class_index]
            str_sec_list = convert_intseclist_to_strseclist(class_result)
            class_result_dic.update({class_key:str_sec_list})
        drone_result_dic[drone_name]=[class_result_dic]
        structured_data[set].append(drone_result_dic)

    structured_data
    return structured_data

def new_answer_list_to_json(list):
    final_output_dic = {"task2_answer":[{}]}
    for set_index, set in enumerate(list):
        temp_dict = {'set_num': set_index+1, 'result':set}
        set_result = organize_result_data(temp_dict)
        final_output_dic['task2_answer'][0].update(set_result)
    return final_output_dic

def output_dict_to_list(task2_result):
    drone_key_list = ['drone_1', 'drone_2', 'drone_3']
    class_key_list = ['M', 'W', 'C']
    answer_list = []
    json_list = task2_result['task2_answer'][0]
    for i, set_num in enumerate(json_list):
        set_list = json_list[set_num]
        set_answer = []
        for j, drone_list in enumerate(set_list):
            class_list = drone_list[drone_key_list[j]]
            drone_answer = []
            for k, class_key in enumerate(class_key_list):
                time_list = class_list[0][class_key]
                time_answer = convert_strseclist_to_intseclist(time_list)
                drone_answer.append(time_answer)
            set_answer.append(drone_answer)
        answer_list.append(set_answer)
    return answer_list

def write_json(result_dic, path='task2.json'):
    with open(path, 'w') as outfile:
        json.dump(result_dic, outfile, indent=2)

def stft(wave, win, mel_fbank):
	stft_cplx = torch.stft(wave[:,:-1], 800, hop_length=200, win_length=800, window=win, center=True, pad_mode='reflect')
	stft_mag = torch.sqrt(stft_cplx[...,0:1]**2 + stft_cplx[...,1:2]**2)[...,0]
	mel_mag = torch.matmul(mel_fbank, stft_mag)
	return mel_mag

def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

def consecutive(data, stepsize=1):
	return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def consecutive_merge(data, threshold=10):
	merge = []
	if len(data) == 1 and len(data[0])==0:
		return np.asarray(merge)
	else:
		temp = [data[0][0], data[0][-1]]
		for interval in data[1:]:
			if temp[-1] + threshold > interval[0]:
				temp[1] = interval[-1]
			else:
				merge.append(temp)
				temp = [interval[0], interval[-1]]
		merge.append(temp)
		return np.asarray(merge)

def seg_to_answer(segment, prob, min_frame=3):
	answer = []
	frame = np.zeros_like(prob)
	for item in segment:
		start, end = item
		if end-start > min_frame:
			weighted_center = (round)((prob[start:end] * np.arange(start, end)).sum() / prob[start:end].sum() / 10)
			answer.append(weighted_center)
			frame[start:end] = 2
			frame[(int)(weighted_center*10)-3:(int)(weighted_center*10)+3] = 1
	if len(answer) == 0:
		answer = [None]
	return np.asarray(answer), frame

def inference(model, audio, win, mel_fbank, window=320, hop=160, gpu=0):
	num_overlap = (window-hop)//8
	audio = audio/(np.max(np.abs(audio))+1e-5)
	overlap = torch.linspace(0, 1, num_overlap).unsqueeze(0).unsqueeze(0).repeat(1,3,1).cuda(gpu)
	wave = torch.from_numpy(audio).unsqueeze(0).cuda(gpu)
	spec = stft(wave, win, mel_fbank).unsqueeze(1)	
	length = spec.size()[-1]
	pad = ((length // hop + 1) * hop) - length
	spec = torch.cat((spec, torch.zeros(1, 1, 80, pad).cuda(gpu)), axis=-1)
	num_iters = (length-window)//hop + 2
	temp_spec = [spec[:, :, :, i*hop:i*hop+window] for i in range(num_iters)]
	temp_spec = torch.cat(temp_spec, axis=0)
	with torch.no_grad():
		logits, recon = model(temp_spec)
		logits_sig_total = torch.sigmoid(logits)
	logits_seq = None
	for i in range(num_iters):
		logits_sig = logits_sig_total[i:i+1, :, :]
		if logits_seq is None:
			logits_seq = logits_sig
		else:
			logits_seq[:,:,-num_overlap:] = (1-overlap) * logits_seq[:,:,-num_overlap:] + overlap * logits_sig[:,:,:num_overlap]
			logits_seq = torch.cat((logits_seq, logits_sig[:,:,num_overlap:]), axis=-1)
	logits_seq = logits_seq[:,:,:(wave.size()[-1]//1600)]
	logits_seq = logits_seq.detach().cpu().numpy()[0]
	m, f, b = logits_seq
	return m, f, b

def postprocessing(m, f, b, threshold=0.8, smooth=5, min_frame=3, merge_frame=10):
	# smoothing
	m_smooth = moving_average(np.pad(m, (smooth//2,smooth//2), mode='edge'), n=smooth)
	f_smooth = moving_average(np.pad(f, (smooth//2,smooth//2), mode='edge'), n=smooth)
	b_smooth = moving_average(np.pad(b, (smooth//2,smooth//2), mode='edge'), n=smooth)
	# threshold
	m_threshold = np.asarray(m_smooth>threshold, dtype='int')
	f_threshold = np.asarray(f_smooth>threshold, dtype='int')
	b_threshold = np.asarray(b_smooth>threshold, dtype='int')
	# index 
	m_index = np.where(m_threshold==1)[0]
	f_index = np.where(f_threshold==1)[0]
	b_index = np.where(b_threshold==1)[0]
	# consecutive segment
	m_consecutive = consecutive(m_index)
	f_consecutive = consecutive(f_index)
	b_consecutive = consecutive(b_index)
	# merge if two segment close
	m_segment = consecutive_merge(m_consecutive, threshold=merge_frame)
	f_segment = consecutive_merge(f_consecutive, threshold=merge_frame)
	b_segment = consecutive_merge(b_consecutive, threshold=merge_frame)
	# middle point
	m_answer, m_frame = seg_to_answer(m_segment, m, min_frame)
	f_answer, f_frame = seg_to_answer(f_segment, f, min_frame)
	b_answer, b_frame = seg_to_answer(b_segment, b, min_frame)
	# plot
	m_answer = np.asarray(m_answer)
	f_answer = np.asarray(f_answer)
	b_answer = np.asarray(b_answer)
	return m_answer, f_answer, b_answer

def cw_metrics_cal(gt, pred):
	correct = 0
	deletion = 0
	insertion = 0
	include_list = []
	for ii in range(len(gt)):
		is_include = 0
		if gt[ii][0] == None:
			if pred[0] == None:
				correct += 1
		else:
			if pred[0] == None:
				deletion += 1
			else:
				gt_start, gt_end = gt[ii][0], gt[ii][1]
				for jj in range(len(pred)):
					if gt_start <= pred[jj] and pred[jj] <= gt_end:
						include_list.append(pred[jj])
						is_include += 1
				if is_include == 0:
					deletion += 1
				elif is_include > 1:
					insertion = is_include - 1
				elif is_include == 1:
					correct += 1
	if pred[0] == None:
		substitution = 0
	else:
		substitution = len(pred) - len(list(set(include_list)))
	return substitution, deletion, insertion, correct

def evaluation(answer_list, set_nums):
	gt_list = []
	gt =[
				[[[187, 191], [194, 197]], [[199, 203]], [[51, 56], [202, 208]]],
				[[[147, 151], [162, 165]], [[154, 158]], [[143, 148]]],
				[[[197, 200], [204, 208]], [[198, 202]], [[210, 215]]],
	]
	for set_num in range(set_nums):
		gt_list.append(gt)
	set_num = len(gt_list)
	total_s, total_d, total_i, total_n, total_correct = 0, 0, 0, 0, 0
	for i in range(set_num):
		sw_s, sw_d, sw_i, sw_n, sw_correct = 0, 0, 0, 0, 0
		for j in range(3): # for drones 1 ~ 3
			dw_s, dw_d, dw_i, dw_n, dw_correct = 0, 0, 0, 0, 0
			for k in range(3): # for class man, woman, child
				cw_s, cw_d, cw_i, cw_correct = cw_metrics_cal(gt_list[i][j][k], answer_list[i][j][k])
				dw_s += cw_s
				dw_d += cw_d
				dw_i += cw_i
				dw_n += len(gt_list[i][j][k])
				dw_correct += cw_correct
			dw_er = (dw_s + dw_d + dw_i) / dw_n
			# print('Set', str(i), 'Drone', str(j), 's, d, i, er, correct:', dw_s, dw_d, dw_i, np.round(dw_er, 2), dw_correct)
			sw_s += dw_s
			sw_d += dw_d
			sw_i += dw_i
			sw_n += dw_n
			sw_er = (sw_s + sw_d + sw_i) / sw_n
			sw_correct += dw_correct
		total_s += sw_s
		total_d += sw_d
		total_i += sw_i
		total_n += sw_n
		total_er = (total_s + total_d + total_i) / total_n
		total_correct += sw_correct
		# print('Subtotal Set', str(i), 's, d, i, er, correct:', sw_s, sw_d, sw_i, np.round(sw_er, 2), sw_correct)
	print('Total', 's, d, i, er, correct:', total_s, total_d, total_i, np.round(total_er, 2), total_correct)
	return total_s, total_d, total_i, total_er, total_correct