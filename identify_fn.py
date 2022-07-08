#This file heavily builds off the detectron2 demo/demo.py file, which has Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import tqdm
import cv2
import json
import numpy as np
import glob
import time

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2_predictor import Detectron2VisualizationDemo
from utils import gt_to_image_format, format_for_alg
from fn_identifier_tools import find_fn_objects, identify_fn_mechanism


def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	# To use demo for Panoptic-DeepLab, please uncomment the following two lines.
	# from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
	# add_panoptic_deeplab_config(cfg)
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	# Set score_threshold for builtin models
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
	cfg.freeze()
	return cfg

def get_parser():
	parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
	parser.add_argument(
		"--config-file",
		default="detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--confidence-threshold",
		type=float,
		default=0.3,
		help="Minimum score for instance predictions to be considered valid",
	)
	parser.add_argument(
		"--input",
		type = str,
		help="Folder creating images to be tested",
	)
	parser.add_argument(
		"--gt",
		type=str,
		required = True,
		help="The location of the annotation file for images being tested. Must be in COCO json format.",
	)
	parser.add_argument(
		"--visFN",
		type=bool,
		default=False,
		help="Do you want to visualise false negatives and their mechanisms",
	)
	parser.add_argument(
		"--vis",
		type=bool,
		default=False,
		help="Do you want to visualise predictions",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	return parser

COCO_CLASSES = np.array(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = get_parser().parse_args()
	setup_logger(name="fvcore")
	logger = setup_logger()

	cfg = setup_cfg(args)

	if 'faster_rcnn' in args.config_file:
		detType = 'FRCNN'
	elif 'retinanet' in args.config_file:
		detType = 'RetNet'
	else:
		print('Only Faster R-CNN and RetinaNet are currently supported. Check that the config file name includes the detector name.')
		exit()

	demo = Detectron2VisualizationDemo(cfg, detType = detType)

	#load in ground-truth annotations
	with open(args.gt, 'r') as f:
		gt_dict = json.load(f)
	
	image_gt_dict = gt_to_image_format(gt_dict, COCO_CLASSES)
	
	all_im_dirs = glob.glob(f'{args.input}*')
	all_fn_mechanisms = []

	for path in tqdm.tqdm(all_im_dirs):
		# use PIL, to be consistent with evaluation
		img = read_image(path, format="BGR")
		predictions, vis_output = demo.run_on_image(img, args.vis)

		if args.vis:
			cv2.namedWindow('Visualised Detections', cv2.WINDOW_NORMAL)
			cv2.imshow('Visualised Detections', vis_output.get_image()[:, :, ::-1])
			if cv2.waitKey(0) == 27:
				cv2.destroyAllWindows() #Esc to quit
				exit()

		#Format image results into dictionary for the FN Mechanisms Algorithm
		im_Results = format_for_alg(predictions, detType)

		#Load gt information
		im_id = int(path.replace(args.input, '').replace('.jpg', '').replace('.png', ''))

		try:
			im_GT = image_gt_dict[im_id]
		except:
			continue #the coco images with no GT objects

		#a list of False negative objects
		fn_objects = find_fn_objects(im_GT, im_Results) 

		#for all fn objects, find their fn mechanism
		im_fn_mechanisms = []		
		for fn_idx, fn_object in enumerate(fn_objects):
			fn_mech = identify_fn_mechanism(fn_object, im_Results, detType)
			im_fn_mechanisms += [fn_mech]

		all_fn_mechanisms += im_fn_mechanisms

	all_fn_mechanisms = np.array(all_fn_mechanisms)


	mechanism_names = ['Proposal Process', 'Regressor', 'Interclass Classification', 'Background Classification', 'Classifier Calibration']
	totalFN = len(all_fn_mechanisms)
	print('Testing with:')
	print(f'    Config: {args.config_file}')
	print(f'    Weights: {args.opts[1]}')
	print(f'	   Image folder: {args.input}')
	print(f'There were {totalErrors} false negatives.')
	for fT, mech in enumerate(all_fn_mechanisms):
		numErrors = np.sum(all_fn_mechanisms == fT)
		print(f'    {mechanism_names[fT]} False Negative Mechanism: {100.*numErrors/totalErrors}% of all false negatives')
				

		

		
