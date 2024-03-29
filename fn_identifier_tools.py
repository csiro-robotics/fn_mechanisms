
##### Author: Dimity Miller, 2022


import numpy as np
import torch
import torch.nn.functional as F

def bbox_iou(gtBox, predBoxes, epsilon=1e-5):
	'''
	Credit to Ronny Restrepo -- http://ronny.rest/tutorials/module/localization_001/iou/
	Inputs:
		gtBox (Tensor) -- Tensor of [x1, y1, x2, y2] describing ground-truth bounding box
		predBoxes (Tensor) -- nx4 Tensor for n predicted bounding boxes, with each row in format [x1, y1, x2, y2]
	Outputs:
		iou (Tensor) -- n Tensor of float values describing IoU between gtBox and n predBoxes

	Description: Calculates the intersection over union between gtBox and all predBoxes.
	'''

	gtBoxes = gtBox.repeat(len(predBoxes), 1)

	x1 = torch.max(torch.cat((gtBoxes[:, 0].unsqueeze(1), predBoxes[:, 0].unsqueeze(1)), dim = 1), dim = 1)[0]
	y1 = torch.max(torch.cat((gtBoxes[:, 1].unsqueeze(1), predBoxes[:, 1].unsqueeze(1)), dim = 1), dim = 1)[0]
	x2 = torch.min(torch.cat((gtBoxes[:, 2].unsqueeze(1), predBoxes[:, 2].unsqueeze(1)), dim = 1), dim = 1)[0]
	y2 = torch.min(torch.cat((gtBoxes[:, 3].unsqueeze(1), predBoxes[:, 3].unsqueeze(1)), dim = 1), dim = 1)[0]

	width = (x2-x1)
	height = (y2-y1)

	width[width < 0] = 0
	height[height < 0] = 0

	area_overlap = width*height

	area_a = (gtBoxes[:, 2] - gtBoxes[:, 0]) * (gtBoxes[:, 3] - gtBoxes[:, 1])
	area_b = (predBoxes[:, 2] - predBoxes[:, 0]) * (predBoxes[:, 3] - predBoxes[:, 1])
	area_combined = area_a + area_b - area_overlap

	iou = area_overlap/ (area_combined + epsilon)

	includedArea = (area_overlap)/area_b

	return iou

def find_fn_objects(gt_dict, pred_dict, iou_thresh = 0.5):
	'''
	Inputs:
		gt_dict (dict) -- dictionary describing gt objects, with keys 'box', 'cls', 'area'. Each key has a list with corresponding information for each gt object described.
		pred_dict (dict) -- dictionary describing predictions, with keys 'output_boxes', 'output_scores', and 'output_classes'. Each key has a list of corresponding information for each prediction.
		iou_thresh (float) -- minimum IoU between a ground truth object and prediction for association to occur
	Outputs:
		fn_objects (list) -- a list of objects not detected by the predictions. Each object is described by [[x1, y1, x2, y2], class, area]

	Description: Given ground-truth objects and detector predictions, it returns information for false negative objects -- objects undetected by the predictions.
	This occurs when there is no prediction with the correct classification and an IoU overlap above iou_thresh
	'''

	fn_objects = []
	
	#no predictions from the detector, all objects are false negatives.
	if 'output_boxes' not in pred_dict.keys():
		for gt_idx in range(len(gt_dict['box'])):
			gt_box = np.array(gt_dict['box'][gt_idx])
			gt_box[2] += gt_box[0]
			gt_box[3] += gt_box[1]
			fn_objects += [[gt_box.tolist(), gt_dict['cls'][gt_idx], gt_dict['area'][gt_idx]]]
		return fn_objects

	###################################################
	### GT Info
	###################################################
	gt_boxes, gt_clses, gt_areas = gt_dict['box'], gt_dict['cls'], gt_dict['area']
	#change gt boxes into [x1, y1, x2, y2] format
	gt_boxes = np.array(gt_boxes)
	gt_boxes[:, 2] += gt_boxes[:, 0]
	gt_boxes[:, 3] += gt_boxes[:, 1]

	###################################################
	### Detections Info
	###################################################    
	boxes = pred_dict['output_boxes']
	scores = pred_dict['output_scores']
	pred_classes = pred_dict['output_classes'].numpy()

	############################################################################################################################################################################
	### Associate detections with GT -- Note, detections not exclusively assigned to GT, i.e. 1 detection may be assigned to multiple GT. Will be addressed in future versions.
	############################################################################################################################################################################    
	det_matches = np.array([i for i in range(len(scores))])

	for gt_idx in range(len(gt_dict['box'])):
		gtBox = gt_boxes[gt_idx].tolist()
		gtCls = gt_clses[gt_idx]
		gtArea = gt_areas[gt_idx]

		#do any detections have at least iou_thresh with GT?
		iou = bbox_iou(torch.Tensor(gtBox), boxes).numpy()
		locMatch = iou >= iou_thresh

		#no detections have localised the GT object
		if np.sum(locMatch) == 0:
			fn_objects += [[gtBox, gtCls, gtArea]]
			continue

		#predicted classes of detections that did localise the GT object
		detIdxes = det_matches[locMatch]
		pCs = np.take(pred_classes, detIdxes)
		#did any detections correctly predict the class? (and have already localised)
		clsMatch = pCs == gtCls

		#no detections predicted the correct GT class
		if np.sum(clsMatch) == 0:
			fn_objects += [[gtBox, gtCls, gtArea]]
			continue

	return fn_objects

def identify_fn_mechanism(fn_object, pred_dict, detType = 'FRCNN', iou_thresh = 0.5, score_thresh = 0.3):
	'''
	Inputs:
		fn_object (list) -- a list of objects not detected by the predictions. Each object is described by [[x1, y1, x2, y2], class, area]
		pred_dict (dict) -- dictionary of information needed for false negative mechanism identification, including keys:
				'box_proposal' (Tensor) -- a tensor of the proposal bounding boxes in format [x1, y1, x2, y2]
				'box_regressed' (Tensor) -- a tensor of the regressed proposal bounding boxes in format [x1, y1, x2, y2]
				if Faster R-CNN: 'logits' (Tensor) -- a tensor of the logit distributions for each proposal (prior to softmax)
				if RetinaNet: 'score_dists' (Tensor) -- a tensor of the sigmoid score distributions for each proposal and each training class
				'nms_inds' (Tensor) -- a tensor of the predictions indices that survive NMS (predictions are proposals after regressed and classified) 
		detType (str) -- 'FRCNN' for Faster R-CNN and 'RetNet' for RetinaNet
		iou_thresh (float) -- minimum IoU between a ground truth object and prediction for association to occur
		score_thresh (float) -- minimum classification score for a detection to be considered valid
	Outputs:
		(int) -- describes the category of false negative mechanism. 0 = proposal process, 1 = regressor, 2 = interclass classification, 3 = background classification, 4 = classifier calibration

	Description: An implementation of Algorithm 1 in the 'What's in the Black Box? The False Negative Mechanisms Inside Object Detectors' paper.
	'''
	gtB, gtC, gtA = fn_object[0], fn_object[1], fn_object[2]

	if detType == 'FRCNN':
		regB, rpnB, logits, nms_inds = pred_dict['box_regressed'], pred_dict['box_proposal'], pred_dict['logits'], pred_dict['nms_inds']
	else:
		regB, rpnB, scores, nms_inds = pred_dict['box_regressed'], pred_dict['box_proposal'], pred_dict['score_dists'], pred_dict['nms_inds']

	##### Check if any regressed boxes meet iou_thresh with GT box (line 1-2 of Alg.1 in paper)
	ious_reg =  bbox_iou(torch.Tensor(gtB), regB)
	mask_reg = ious_reg >= iou_thresh
	
	if torch.sum(mask_reg) != 0:
		#there was a regressed box that localised object. Check if the ground-truth class score met the score cut-off of the detector (line 3-4 in Alg. 1)
		if detType == 'FRCNN':
			box_logits = logits[mask_reg]
			mask_scores = F.softmax(box_logits, dim = -1) #softmax scores from logits
		else:
			mask_scores = scores[mask_reg]

		mask_gt_scores = mask_scores[:, gtC] >= score_thresh

		if torch.sum(mask_gt_scores) != 0:
			#there was a localised detection with a correct classification above score threshold. Must have been suppressed by NMS. (line 5 in Alg.1)
			return 4 #classifier calibration false negative mechanism

		#check if any other training classes had a score above the score cut-off (lines 6 in Alg.1)
		if detType == 'FRCNN':
			mask_interclass_scores = mask_scores[:, :-1] >= score_thresh #omit background class
		else:
			mask_interclass_scores = mask_scores >= score_thresh #retinanet doesn't have explicit background class

		if torch.sum(mask_interclass_scores) != 0:
			#there was a localised detection with a misclassification. (line 7 in Alg.1)
			return 2 #interclass classification false negative mechanism

		#if all localised detections didn't have a training class score above score cut-off, this means the detection was misclassified as background (line 8-9 in Alg.1)
		return 3 #background classification false negative mechanism

	else:
		#there was not a regressed box that localised the object. check if there was a proposal that localised the object. (lines 10-12 in Alg.1)
		ious_prop =  bbox_iou(torch.Tensor(gtB), rpnB)
		mask_prop = ious_prop >= iou_thresh
		if torch.sum(mask_prop) != 0:
			#there was at least 1 object proposal that localised the object. Must be failure of regressor (lines 12-13)
			return 1 #regressor false negative mechanism
		else:
			#there were no object proposals that localised the object. (lines 14-15)
			return 0 #proposal process false negative mechanism