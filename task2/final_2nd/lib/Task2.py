import torch
import torchaudio

import os
import sys
import numpy as np
import julius


from task2.final_2nd.lib.marg_utils import *

def task2_inference(data_path, set_nums):
	gpu = 0
	# post processing
	threshold = 0.2
	smooth = 13
	min_frame = 18

	module = __import__('task2.final_2nd.lib.marg_model', fromlist=[''])
	generator = module.model()
	generator.to('cuda:{}'.format(gpu))
	generator.load_state_dict(torch.load('task2/final_2nd/weights/model.pt', map_location='cuda:{}'.format(gpu)))
	generator.eval()

	win = np.load('task2/final_2nd/lib/window.npy')
	win = torch.from_numpy(win).cuda(gpu)
	mel_fbank = np.load('task2/final_2nd/lib/mel_fbank.npy')
	mel_fbank = torch.from_numpy(mel_fbank).cuda(gpu)

	drone_num = 3
	answer_list = []
	for i in range(set_nums):
		sub_answer_list = []
		for j in range(drone_num):
			folder_name = 'set_0' + str(i+1)
			# file_name = 'set0' + str(i+1) + '_drone0' + str(j+1) + '_ch1.wav'
			# audio_48k, sr = torchaudio.load(os.path.join(data_path, folder_name, file_name))					

			audio_48k = 0
			for k in range(2):
				file_name = 'set0' + str(i+1) + '_drone0' + str(j+1) + '_ch' + str(k+1) + '.wav'
				audio_48ks, sr = torchaudio.load(os.path.join(data_path, folder_name, file_name))					
				audio_48k += audio_48ks
			audio_48k /= 2

			audio_16k = julius.resample_frac(audio_48k, sr, 16000).numpy()[0]
			with torch.no_grad():
				m, f, b = inference(generator, audio_16k, win, mel_fbank, gpu=gpu)
				mf, ff, bf = postprocessing(m, f, b, threshold=threshold, smooth=smooth, min_frame=min_frame)
			sub_answer_list.append([list(mf), list(ff), list(bf)])
		answer_list.append(sub_answer_list)
	out_str = new_answer_list_to_json(answer_list)
	# print(answer_list)
	# return answer_list, out_str
	return out_str

# 	audio1, sr = librosa.load('/home/home/juheon/gc_2021/validation_data_new/set01_drone01_mono_16k.wav', sr=None, mono=True)
# 	audio2, sr = librosa.load('/home/home/juheon/gc_2021/validation_data_new/set01_drone02_mono_16k.wav', sr=None, mono=True)
# 	audio3, sr = librosa.load('/home/home/juheon/gc_2021/validation_data_new/set01_drone03_mono_16k.wav', sr=None, mono=True)

# # model inference
# with torch.no_grad():
# 	m1, f1, b1 = inference(generator, audio1, win, mel_fbank, gpu=gpu)
# 	m2, f2, b2 = inference(generator, audio2, win, mel_fbank, gpu=gpu)
# 	m3, f3, b3 = inference(generator, audio3, win, mel_fbank, gpu=gpu)



# m1f, f1f, b1f = postprocessing(m1, f1, b1, threshold=threshold, smooth=smooth, min_frame=min_frame)
# m2f, f2f, b2f = postprocessing(m2, f2, b2, threshold=threshold, smooth=smooth, min_frame=min_frame)
# m3f, f3f, b3f = postprocessing(m3, f3, b3, threshold=threshold, smooth=smooth, min_frame=min_frame)

# # write answer
# answer_list = [
# 	[
# 		[list(m1f), list(f1f), list(b1f)],
# 		[list(m2f), list(f2f), list(b2f)],
# 		[list(m3f), list(f3f), list(b3f)],
# 	],
# ]

# # validation
# gt_list = [ 
# 	[
# 		[[[187, 191], [194, 197]], [[199, 203]], [[51, 56], [202, 208]]],
# 		[[[147, 151], [162, 165]], [[154, 158]], [[143, 148]]],
# 		[[[197, 200], [204, 208]], [[198, 202]], [[210, 215]]],
# 	],
# ]
# s, d, i, er, correct = evaluation(gt_list, answer_list)


