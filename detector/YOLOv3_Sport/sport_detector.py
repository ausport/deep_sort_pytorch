import torch
import sys
# sys.path.insert(0, 'YOLOv3_Sport')
from .models import *
from .helpers.utils import *
from .helpers.datasets import *
import numpy as np
import cv2

from .models import *
from .helpers.utils import *
from .helpers.datasets import *

def xyxy_to_xywh(boxes_xyxy):
	if isinstance(boxes_xyxy, torch.Tensor):
		boxes_xywh = boxes_xyxy.clone()
	elif isinstance(boxes_xyxy, np.ndarray):
		boxes_xywh = boxes_xyxy.copy()

	boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
	boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
	boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
	boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

	return boxes_xywh

class YOLOv3_4_SPORT(object):
	def __init__(self, cfgfile, weightfile, namesfile, score_thresh=0.7, conf_thresh=0.01, nms_thresh=0.45,
	             is_xywh=False, use_cuda=True):

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		print(cfgfile)
		self.model_def = cfgfile
		self.weights_path = weightfile
		self.img_size = 608

		# Set up model
		self.net = Darknet(self.model_def, img_size=self.img_size).to(device)

		print('Loading weights from %s... Done!' % weightfile)
		self.device = "cuda" if use_cuda else "cpu"
		if self.weights_path.endswith(".weights"):
			# Load darknet weights
			self.net.load_darknet_weights(self.weights_path)
		else:
			# Load checkpoint weights
			self.net.load_state_dict(torch.load(self.weights_path))

		self.net.eval()  # Set in evaluation mode
		self.net.to(self.device)
	#
	# 	# constants
		self.size = 608, 608
		self.score_thresh = score_thresh
		self.conf_thresh = conf_thresh
		self.nms_thresh = nms_thresh
		self.use_cuda = use_cuda
		self.num_classes = 1
		self.class_names = self.load_class_names(namesfile)
	#
	def __call__(self, ori_img):
		# img to tensor
		assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
		img = ori_img.astype(np.float) / 255.

		img = cv2.resize(img, self.size)
		img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

		# Get detections
		with torch.no_grad():
			img = img.to(self.device)
			detections = self.net(img)
			boxes = non_max_suppression(detections, self.conf_thresh, self.nms_thresh)[0].cpu()
			boxes = boxes[boxes[:, -2] > self.score_thresh, :]  # bbox xmin ymin xmax ymax

		if len(boxes) == 0:
			return None, None, None

		height, width = ori_img.shape[:2]
		height /= 608
		width /= 608

		bbox = boxes[:, :4]
		bbox = xyxy_to_xywh(bbox)
		bbox = bbox * torch.FloatTensor([[width, height, width, height]])
		cls_conf = boxes[:, 5]
		cls_ids = boxes[:, 6].long()
		return bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()

	def load_class_names(self, namesfile):
		with open(namesfile, 'r', encoding='utf8') as fp:
			class_names = [line.strip() for line in fp.readlines()]
		return ['player']