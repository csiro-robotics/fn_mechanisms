###### Author Dimity Miller, 2022.

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from fn_identifier_tools import bbox_iou
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def gt_to_image_format(gt_dict, class_list):
	image_dict = {}

	annotations = gt_dict['annotations']
	categories = gt_dict['categories']
	images = gt_dict['images']

	clsDict = {}
	for clsInfo in categories:
		clsId = np.where(class_list == clsInfo['name'])[0][0]
		clsDict[clsInfo['id']] = clsId

	totalGT = 0

	allAreas = []
	for anno in annotations:
		if anno['iscrowd']:
			continue

		if anno['image_id'] not in image_dict.keys(): #if no existing field for this image
			image_dict[anno['image_id']] = {}
			image_dict[anno['image_id']]['box'] = []
			image_dict[anno['image_id']]['cls'] = []
			image_dict[anno['image_id']]['area'] = []
		
		box = anno['bbox']
		area = anno['area']
		class_id = clsDict[anno['category_id']]

		image_dict[anno['image_id']]['box'] += [box]
		image_dict[anno['image_id']]['cls'] += [class_id]
		image_dict[anno['image_id']]['area']+= [area]
		totalGT += 1

	print(f"Loaded {len(image_dict.keys())} image annotations and {totalGT} total annotations.")
	
	return image_dict

def format_for_alg(prediction_dict, dType = 'FRCNN'):
	results = {}

	if len(prediction_dict['instances']) != 0:
		results['output_scores'] = prediction_dict['instances'].scores.cpu().detach()
		results['output_classes'] =  prediction_dict['instances'].pred_classes.cpu().detach()
		results['output_boxes'] =  prediction_dict['instances'].pred_boxes.tensor.cpu().detach()

	results['box_regressed'] = prediction_dict['reg_box']
	results['box_proposal'] = prediction_dict['prop_box']
	results['nms_inds'] = prediction_dict['pred_inds'][0]

	if dType == 'FRCNN':
		results['logits'] = prediction_dict['logits']
	else:
		results['score_dists'] = prediction_dict['score_dists']

	return results


def draw_fn_mechanism(detType, cfg, image, fn_object, im_dict, fn_mech, class_list, iou_thresh = 0.5, score_thresh = 0.3):
	image = image[:, :, ::-1]

	metadata = MetadataCatalog.get(
			cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
		)

	fn_im = draw_fn(image, metadata, fn_object, class_list)

	if fn_mech == 0 or fn_mech == 1:
		if fn_mech == 0: #proposal process false negative mechanism. Draw proposals that have some overlap but less than needed, color coded by localisation.
			all_boxes = im_dict['box_proposal']

		if fn_mech == 1: #regressor false negative mechanism. Draw regressed boxes that have some overlap but less than needed, color coded by localisation.
			all_boxes = im_dict['box_regressed']
		
		ious =  bbox_iou(torch.Tensor(fn_object[0]), all_boxes).detach().numpy()
		iou_mask = ious > 0
		boxes = all_boxes[iou_mask].detach().numpy()
		order = np.argsort(ious[iou_mask])

		ordered_boxes = boxes[order] #most overlap last (makes visualisation easier)

		max_iou = np.max(ious)

		colormap = plt.cm.get_cmap('plasma')
		cNorm = colors.Normalize(vmin = 0, vmax = 0.5)
		scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap = colormap)

		colorList = [scalarMap.to_rgba(x) for x in ious[iou_mask][order]]
		mech_im = draw_proposals(image, metadata, ordered_boxes, colorList, max_iou, fn_mech)

	elif fn_mech == 2: #interclass classification false negative mechanism. Draw most confident interclass misclassification that localised object.
		reg_boxes = im_dict['box_regressed']
		ious_reg =  bbox_iou(torch.Tensor(fn_object[0]), reg_boxes)
		ious_reg_mask = ious_reg >= iou_thresh
		
		if detType == 'FRCNN':
			box_logits = im_dict['logits'][ious_reg_mask]
			mask_scores = F.softmax(box_logits, dim = -1)[:, :-1] #softmax scores from logits, don't include background
		else:
			mask_scores = im_dict['score_dists'][ious_reg_mask]
		
		cls_max, cls_idx = torch.max(mask_scores, dim = 1)
		score_max, box_idx = torch.max(cls_max), torch.argmax(cls_max)

		box = np.array([reg_boxes[ious_reg_mask][box_idx].detach().numpy()])
		lbl = [f'{class_list[cls_idx[box_idx]]}: {score_max:.2f},\n Iou: {ious_reg[ious_reg_mask][box_idx]:.2f}']
		
		mech_im = draw_mechanism_classifier(image, metadata, box, lbl, ['b'], fn_mech)

	elif fn_mech == 3: #background classification mechanism. Draw best fitting box that was classified as background
		#find best fitting box classified as background
		reg_boxes = im_dict['box_regressed']
		ious_reg =  bbox_iou(torch.Tensor(fn_object[0]), reg_boxes)
		best_box_idx = torch.argmax(ious_reg)
		best_box = np.array([reg_boxes[best_box_idx].detach().numpy()])

		#find the background score if it was faster r-cnn
		if detType == 'FRCNN':
			box_logit = im_dict['logits'][best_box_idx]
			scores = F.softmax(box_logit, dim = -1) #softmax scores from logits
			bkg_score = scores[-1]
			lbl = [f'Bkg: {bkg_score:.2f},\n IoU: {ious_reg[best_box_idx]:.2f}']
		else:
			#retinanet has no explicit background class, so just label as background
			lbl = ['Bkg,\n IoU: {ious_reg[best_box_idx]:.2f}']

		mech_im = draw_mechanism_classifier(image, metadata, best_box, lbl, ['b'], fn_mech)

	elif fn_mech == 4: #classifier calibration false negative mechanism. Draw the correct detection (localised and most confident) that was suppressed and the detection that suppressed it.
		reg_boxes = im_dict['box_regressed']
		ious_reg =  bbox_iou(torch.Tensor(fn_object[0]), reg_boxes)
		ious_mask = ious_reg >= iou_thresh

		if detType == 'FRCNN':
			box_logits = im_dict['logits'][ious_mask]
			gt_scores = F.softmax(box_logits, dim = -1)[:, fn_object[1]] #softmax scores from logits, don't include background
		else:
			gt_scores = im_dict['score_dists'][ious_mask][:, fn_object[1]]

		#potential correct box that was suppressed
		most_confident, most_confident_idx = torch.max(gt_scores), torch.argmax(gt_scores)

		correct_box = reg_boxes[ious_mask][most_confident_idx].detach().numpy()
		correct_lbl = f'{class_list[fn_object[1]]}: {most_confident:.2f}, IoU: {ious_reg[ious_mask][most_confident_idx]:.2f}'

		#output box that suppressed it
		pred_clses = im_dict['output_classes']
		pred_boxes = im_dict['output_boxes']
		pred_scores = im_dict['output_scores']

		cls_mask = pred_clses == fn_object[1]
		cls_boxes = pred_boxes[cls_mask]

		cls_ious = bbox_iou(torch.Tensor(fn_object[0]), cls_boxes)
		most_overlap = torch.argmax(cls_ious)
		
		incorrect_box = cls_boxes[most_overlap].detach().numpy()
		incorrect_score = pred_scores[cls_mask][most_overlap]
		incorrect_lbl = f'{class_list[fn_object[1]]}: {incorrect_score:.2f}, IoU: {cls_ious[most_overlap]:.2f}'

		mech_im = draw_mechanism_calibration(image, metadata, np.array([correct_box, incorrect_box]), [correct_lbl, incorrect_lbl], ['g', 'r'])
		
	else:
		print('Not a recognised type of false negative mechanism.')
		exit()

	return fn_im, mech_im

def draw_proposals(image, metadata, boxes, box_colors, max_iou, mech_type):
	visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

	for i, box in enumerate(boxes):
		color = box_colors[i]
		visualizer.draw_box(box, edge_color=color)

	if mech_type == 0:
		lbl = 'Proposal Process False Negative Mechanism'
	if mech_type == 1:
		lbl = 'Regressor False Negative Mechanism'
	visualizer.draw_text(lbl, (0, 0), horizontal_alignment = 'left', color = 'w')

	lbl = f'Maximum IoU: {max_iou:.3f}'
	im_height, im_width, _ = image.shape
	txt_pos = (int(im_width*0.5), int(im_height*0.1))

	output = visualizer.draw_text(lbl, txt_pos, color = 'w').get_image()[:, :, ::-1]

	return output

def draw_mechanism_calibration(image, metadata, bbox, label, box_colors):
	visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
	visualizer.overlay_instances(boxes = bbox, assigned_colors = box_colors)

	if bbox[0, 0] < image.shape[1] * 0.3 and bbox[0, 2] < image.shape[1] * 0.7:
		visualizer.draw_text(label[0], (bbox[0, 2], bbox[0, 1]), color = box_colors[0], horizontal_alignment = 'left')
		visualizer.draw_text(label[1], (bbox[1, 2], bbox[1, 3]), color = box_colors[1], horizontal_alignment = 'left')
	
	else:
		visualizer.draw_text(label[0], (bbox[0, 0], bbox[0, 1]), color = box_colors[0], horizontal_alignment = 'right')
		visualizer.draw_text(label[1], (bbox[1, 0], bbox[1, 3]), color = box_colors[1], horizontal_alignment = 'right')

	output = visualizer.draw_text('Classifier Calibration False Negative Mechanism', (0, 0), color = 'w', horizontal_alignment = 'left').get_image()[:, :, ::-1]

	return output

def draw_mechanism_classifier(image, metadata, bbox, label, box_colors, mech_type):
	visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
	visualizer.overlay_instances(boxes = bbox, labels = label, assigned_colors = box_colors)

	if mech_type == 2:
		lbl = 'Interclass Classification False Negative Mechanism'
	if mech_type == 3:
		lbl = 'Background Classification False Negative Mechanism'

	output = visualizer.draw_text(lbl, (0, 0), color = 'w', horizontal_alignment = 'left').get_image()[:, :, ::-1]

	return output

def draw_fn(image, metadata, fn_object, class_list):
	visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
	
	bbox =np.array([fn_object[0]])
	label = [class_list[fn_object[1]]]
	
	visualizer.overlay_instances(boxes = bbox, labels = label, assigned_colors = ['r'])
	output = visualizer.draw_text('Ground Truth of False Negative Object', (0, 0), color = 'w', horizontal_alignment = 'left').get_image()[:, :, ::-1]

	return output