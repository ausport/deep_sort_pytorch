import torch
import sys
sys.path.insert(0, 'YOLOv3_Sport')
from .models import *
from .utils.utils import *
from .utils.datasets import *
import numpy as np
import cv2

from darknet import Darknet
from yolo_utils import get_all_boxes, nms, post_process, xywh_to_xyxy, xyxy_to_xywh
from nms import boxes_nms


class YOLOv3_4_SPORT(object):
	def __init__(self, cfgfile, weightfile, namesfile, score_thresh=0.7, conf_thresh=0.01, nms_thresh=0.45,
	             is_xywh=False, use_cuda=True):

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print(cfgfile)
		self.model_def = "config/yolov3-custom.cfg"
		self.weights_path = weightfile
		self.img_size = 608

		# Set up model
		model = Darknet(self.model_def, img_size=self.img_size).to(device)

		if self.weights_path.endswith(".weights"):
			# Load darknet weights
			model.load_darknet_weights(self.weights_path)
		else:
			# Load checkpoint weights
			model.load_state_dict(torch.load(self.weights_path))

		model.eval()  # Set in evaluation mode
		sys.exit(1)
	# 	# net definition
	# 	self.net = Darknet(cfgfile)
	# 	self.net.load_weights(weightfile)
	# 	print('Loading weights from %s... Done!' % (weightfile))
	# 	self.device = "cuda" if use_cuda else "cpu"
	# 	self.net.eval()
	# 	self.net.to(self.device)
	#
	# 	# constants
	# 	self.size = self.net.width, self.net.height
	# 	self.score_thresh = score_thresh
	# 	self.conf_thresh = conf_thresh
	# 	self.nms_thresh = nms_thresh
	# 	self.use_cuda = use_cuda
	# 	self.is_xywh = is_xywh
	# 	self.num_classes = self.net.num_classes
		self.class_names = self.load_class_names(namesfile)
	#
	# def __call__(self, ori_img):
	# 	# img to tensor
	# 	assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
	# 	img = ori_img.astype(np.float) / 255.
	#
	# 	img = cv2.resize(img, self.size)
	# 	img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
	#
	# 	# forward
	# 	with torch.no_grad():
	# 		img = img.to(self.device)
	# 		out_boxes = self.net(img)
	# 		boxes = get_all_boxes(out_boxes, self.conf_thresh, self.num_classes,
	# 		                      use_cuda=self.use_cuda)  # batch size is 1
	# 		# boxes = nms(boxes, self.nms_thresh)
	#
	# 		boxes = post_process(boxes, self.net.num_classes, self.conf_thresh, self.nms_thresh)[0].cpu()
	# 		boxes = boxes[boxes[:, -2] > self.score_thresh, :]  # bbox xmin ymin xmax ymax
	#
	# 	if len(boxes) == 0:
	# 		return None, None, None
	#
	# 	height, width = ori_img.shape[:2]
	# 	bbox = boxes[:, :4]
	# 	if self.is_xywh:
	# 		# bbox x y w h
	# 		bbox = xyxy_to_xywh(bbox)
	#
	# 	bbox = bbox * torch.FloatTensor([[width, height, width, height]])
	# 	cls_conf = boxes[:, 5]
	# 	cls_ids = boxes[:, 6].long()
	# 	return bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()

	def load_class_names(self, namesfile):
		with open(namesfile, 'r', encoding='utf8') as fp:
			class_names = [line.strip() for line in fp.readlines()]
		return ['player']