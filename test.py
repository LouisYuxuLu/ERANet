# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:48:05 2020

@author: Administrator
"""


import torch
import torch.nn as nn


import numpy as np
import cv2
import time
import os
from ERANet import *
from thop import profile

from fvcore.nn import FlopCountAnalysis, parameter_count_table

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_checkpoint(checkpoint_dir,Type):
    
	if Type == 0:
		Rep ='Rain'
	elif Type == 1:
		Rep ='Low'
	elif Type == 2:
		Rep ='Haze'
	else:
		Rep = 'All'

        
	model_info = torch.load(checkpoint_dir + Rep +'_checkpoint.pth.tar')
	net = Main()
	device_ids = [0]
	model = nn.DataParallel(net, device_ids=device_ids).cuda()
	model.load_state_dict(model_info['state_dict'])

	return model


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	

if __name__ == '__main__':
	Type = 0   #Rain is 0, LowLight is 1, Haze is 2
	if Type == 0:
		test_dir = './input/rainy'
		result_dir = './output/rainy'
	elif Type == 1:
		test_dir = './input/low'
		result_dir = './output/low'
	elif Type == 2:
		test_dir = './input/hazy'
		result_dir = './output/hazy'
	checkpoint_dir = './checkpoint/'
	testfiles = os.listdir(test_dir)

	model = load_checkpoint(checkpoint_dir,Type)

 

	for f in range(len(testfiles)):
		model.eval()
		with torch.no_grad():
			img_c = cv2.imread(test_dir + '/' + testfiles[f]) / 255.0
			w,h,c= img_c.shape
			img_l = hwc_to_chw(np.array(img_c).astype('float32'))
			input_var = torch.from_numpy(img_l.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
			s = time.time()
			E_out = model(input_var)            
			e = time.time()   
			print(input_var.shape)       
			print('Time:%.4f'%(e-s))    
			E_out = chw_to_hwc(E_out.squeeze().cpu().detach().numpy())			               
			cv2.imwrite(result_dir + '/' + testfiles[f][:-4] + '_ERANet.png',np.clip(E_out*255,0.0,255.0))

 