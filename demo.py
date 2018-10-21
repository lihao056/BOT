import cv2
import torch
from PIL import Image
import time

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchreid import transforms as T
from torchreid import data_manager
from torchreid.dataset_loader import ImageDataset_demo
from torchreid import models
import os
import argparse

parser = argparse.ArgumentParser(description='Age recognition:video and image ')
parser.add_argument('-a', '--arch', type=str, default='resnet50_chart', choices=models.get_names())
parser.add_argument('--resume', type=str, default='log/resnet_50_chart/best_model.pth.tar', metavar='PATH')
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

num2label = {'0':'AreaGraph',
             '1':'BarGraph',
             '2':'LineGraph',
             '3':'Map',
             '4':'ParetoChart',
             '5':'PieChart',
             '6':'RadarPlot',
             '7':'ScatterGraph',
             '8':'Table',
             '9':'VennDiagram'}

'''
输入：
	1）model_dir:模型存放地址（pytorch）
	2）arch:base模型类别，如resnet18,resnet50等

'''
def Init_chart_model(model_dir,arch):
	CUDA = torch.cuda.is_available()
	print("Initializing model: {}".format(arch))
	model_chart = models.init_model(name=arch, num_classes=10, loss={'xent'}, use_gpu=CUDA)
	if model_dir:
		print("Loading checkpoint from '{}'".format(model_dir))
		checkpoint = torch.load(model_dir)
		model_chart.load_state_dict(checkpoint['state_dict'])
		print("Age Network successfully loaded")
	if CUDA:
		model_chart = nn.DataParallel(model_chart).cuda()
	return model_chart

'''
输入：
	1）model_chart:性别识别模型（pytorch）
	2）img:图表图片
输出：
	kind:预测的图标类型
'''
def chart_recognition(model_chart,img_dir):
	# img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	transform = T.Compose([
		T.Resize((256, 128)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	loader = DataLoader(
        ImageDataset_demo(img_dir, transform=transform),
		batch_size=1, shuffle=False, num_workers=0,
		pin_memory=True, drop_last=False, )
	model_chart.eval()
	with torch.no_grad():
		for batch_idx, img2 in enumerate(loader):
			if torch.cuda.is_available(): img2 = img2.cuda()
			score = model_chart(img2)
			print(score)
			chart = torch.argmax(score.data, 1)
			chart = chart[0].cpu().numpy()
			print(chart)
			kind = num2label[str(chart)]



	return kind

'''
识别图表类型
输入：1）test_img_dir:测试图片的地址
'''
def recognition(test_img_dir,resume,arch):
	model_chart = Init_chart_model(resume,arch)
	# img = Image.open(test_img_dir).convert('RGB')
	# img = cv2.imread(test_img_dir)
	kind = chart_recognition(model_chart,test_img_dir)

	return kind

if __name__ == '__main__':
	test_img_dir = './7.jpg'
	kind  = recognition(test_img_dir,args.resume,args.arch)
	print('图表类型为：%s'%kind)

