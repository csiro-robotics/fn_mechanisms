#This file heavily builds off the detectron2 demo/predictor.py file, which has Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.structures import Boxes, Instances

from detectron2.layers import batched_nms
from detectron2.modeling.postprocessing import detector_postprocess


class Detectron2VisualizationDemo(object):
	def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, detType = 'FRCNN'):
		"""
		Args:
			cfg (CfgNode):
			instance_mode (ColorMode):
			parallel (bool): whether to run the model in different processes from visualization.
				Useful since the visualization logic can be slow.
		"""
		self.metadata = MetadataCatalog.get(
			cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
		)
		self.cpu_device = torch.device("cpu")
		self.instance_mode = instance_mode

		self.parallel = parallel
		if parallel:
			num_gpu = torch.cuda.device_count()
			self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
		else:
			self.predictor = DefaultPredictor(cfg)
			
		self.detType = detType

	def run_on_image(self, image_orig, vis = False):
		"""
		Args:
			image_orig (np.ndarray): an image of shape (H, W, C) (in BGR order).
				This is the format used by OpenCV.

		Returns:
			predictions (dict): the output of the model.
		"""

		image = image_orig.copy()		
		if self.predictor.input_format == "RGB":
			# whether the model expects BGR inputs or RGB
			image = image_orig[:, :, ::-1]

		height, width = image.shape[:2]
		image = self.predictor.aug.get_transform(image).apply_image(image)

		image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

		inputs = [{"image": image, "height": height, "width": width}]

		images = self.predictor.model.preprocess_image(inputs)

		#for when we need to reshape boxes back to image size
		origImHeight = height
		origImWidth = width
		newImHeight = images[0].size(1)
		newImWidth = images[0].size(2)
		heightConvert = origImHeight/newImHeight
		widthConvert = origImWidth/newImWidth
		box_resizer = torch.Tensor([widthConvert, heightConvert, widthConvert, heightConvert]).cuda()

		##################################################################
		## STEP A in detection pipeline -- extract features with backbone
		##################################################################
		features = self.predictor.model.backbone(images.tensor)  # set of cnn features across different scales, each scale is 256xsomething1xsomething2

		#Faster R-CNN specific 
		if self.detType == 'FRCNN':
			##################################################################
			## STEP B in detection pipeline -- collect object proposals from Faster R-CNN
			##################################################################
			proposals, _ = self.predictor.model.proposal_generator(images, features, None)  # RPN, generates a set of 1000 proposal bboxes and their objectness logit

			features_ = [features[f] for f in self.predictor.model.roi_heads.box_in_features] #collects the features that are roi in features (leaves out p6?)

			box_features = self.predictor.model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals]) #taking in features and proposed bboxes and outputting 1000x256x7x7

			box_features = self.predictor.model.roi_heads.box_head(box_features)  # features of all 1k candidates in a 1024d vector

			##################################################################
			## STEP C in detection pipeline -- regress box offsets and logits for classification
			##################################################################
			predictions = self.predictor.model.roi_heads.box_predictor(box_features)   #returns 1000x81 logits and 1000x320 bbox predictions
		
			##################################################################
			## STEP D in detection pipeline -- NMS
			##################################################################

			pred_instances, pred_inds = self.predictor.model.roi_heads.box_predictor.inference(predictions, proposals) #does NMS
			pred_instances = self.predictor.model.roi_heads.forward_with_given_boxes(features, pred_instances) #doesn't do anything 

			#scale final (after NMS) box predictions also back to image size										 
			predictions_final = self.predictor.model._postprocess(pred_instances, inputs, images.image_sizes)[0]  # scale box to orig size

			##################################################################
			## Grab things we need for our algorithm
			##################################################################
			
			distributions = predictions[0] #before NMS, this is the raw classification logits 
			all_boxes_reg = self.predictor.model.roi_heads.box_predictor.predict_boxes(predictions, proposals)[0] ### this takes the predicted box offsets and calculates the overall predicted boxes

			#process regressed boxes back to readable format
			image_shape = [x.image_size for x in proposals][0]
			num_bbox_reg_classes = all_boxes_reg.shape[1] // 4
			all_boxes_reg = Boxes(all_boxes_reg.reshape(-1, 4))
			all_boxes_reg.clip(image_shape)
			all_boxes_reg = all_boxes_reg.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

			#only take regressed box for predicted class from the classifier
			all_preds = distributions[:, :-1]
			pred_cls = torch.argmax(all_preds, dim = 1).unsqueeze(-1)
			pred_idxes = pred_cls.repeat(1, 4).unsqueeze(1)
			boxes_reg = torch.gather(all_boxes_reg, 1, pred_idxes).squeeze()

			#resize the RPN boxes and the regressed boxes back to original image size
			boxes_reg_resized = boxes_reg * box_resizer
			boxes_rpn = proposals[0].proposal_boxes.tensor
			boxes_rpn_resized = boxes_rpn * box_resizer


			#return relevant information for algorithm
			# predictions_final['rpnObject'] = proposals[0].objectness_logits #if you want, here is how to grab the RPN objectness logits predicted for each anchor. We don't use this, so commenting out.
			predictions_final['prop_box'] = boxes_rpn_resized.cpu()
			predictions_final['pred_inds'] = pred_inds #the proposals that survive NMS
			predictions_final['reg_box'] = boxes_reg_resized.cpu() 
			predictions_final['logits'] = distributions.cpu()
		
		#RetinaNet specific
		else:
			##################################################################
			## STEP B in detection pipeline -- collect anchors from RetinaNet
			##################################################################
			features = [features[f] for f in self.predictor.model.head_in_features] #different size feature maps 256xnxm and there are 5 of them
			anchors = self.predictor.model.anchor_generator(features)

			##################################################################
			## STEP C in detection pipeline -- regress box offsets and logits for classification
			##################################################################
			logits = []
			bbox_reg = []
			for feature in features:
				logits.append(self.predictor.model.head.cls_score(self.predictor.model.head.cls_subnet(feature)))
				bbox_reg.append(self.predictor.model.head.bbox_pred(self.predictor.model.head.bbox_subnet(feature)))

			pred_logits, pred_anchor_deltas = self.predictor.model._transpose_dense_predictions((logits, bbox_reg), [self.predictor.model.num_classes, 4]) #process results into correct format
	
			image_size = images.image_sizes[0] #check
			scores_per_image = [x[0].sigmoid_().detach() for x in pred_logits]
			deltas_per_image = [x[0] for x in pred_anchor_deltas]

			pred = self.predictor.model._decode_multi_level_predictions(  #turns into prediction Instances with boxes, scores and predicted classes (does the same as above plus more, but removes score distributions to single score)
				anchors,
				scores_per_image,
				deltas_per_image,
				self.predictor.model.test_score_thresh,
				self.predictor.model.test_topk_candidates,
				image_size,
			)

			##################################################################
			## STEP D in detection pipeline -- NMS
			##################################################################
			pred_inds = batched_nms(  # per-class NMS
				pred.pred_boxes.tensor, pred.scores, pred.pred_classes, self.predictor.model.test_nms_thresh
			)
			results_per_image = pred[pred_inds[: self.predictor.model.max_detections_per_image]]
			
			im_height = inputs[0].get("height", image_size[0])
			im_width = inputs[0].get("width", image_size[1])
			final_detections = detector_postprocess(results_per_image, im_height, im_width)
			predictions_final = {"instances": final_detections}

			##################################################################
			## Grab things we need for our algorithm
			##################################################################
			#grab each regressed box at each anchor level and size to image size as well as anchors and reshape to image size
			num_anchors = 0
			for anch_level in range(len(anchors)):
				num_anchors += len(anchors[anch_level])

			boxes_reg_resized = torch.zeros(num_anchors,4)
			multilevel_anchors = torch.zeros(num_anchors,4)
			all_scores = torch.zeros(num_anchors,self.predictor.model.num_classes)
			current_idx = 0
			for anch_level in range(len(anchors)):
				box_level = self.predictor.model.box2box_transform.apply_deltas(
					deltas_per_image[anch_level], anchors[anch_level].tensor
				)	
				box_level_resized = box_level * box_resizer
				anchor_resized = anchors[anch_level].tensor * box_resizer

				end_idx = len(box_level_resized)+current_idx

				boxes_reg_resized[current_idx:end_idx] = box_level_resized.cpu()
				multilevel_anchors[current_idx:end_idx] = anchor_resized.cpu()
				all_scores[current_idx:end_idx] = scores_per_image[anch_level].cpu()

				current_idx = end_idx


			# #return relevant information for algorithm
			predictions_final['prop_box'] = multilevel_anchors
			predictions_final['reg_box'] = boxes_reg_resized
			predictions_final['score_dists'] = all_scores
			predictions_final['pred_inds'] = [pred_inds] #the proposals that survive NMS

		image_orig = image_orig[:, :, ::-1]
		visualizer = Visualizer(image_orig, self.metadata, instance_mode=self.instance_mode)
		if "instances" in predictions_final:
			instances = predictions_final["instances"].to(self.cpu_device)
			vis_output = visualizer.draw_instance_predictions(predictions=instances)
		else:
			vis_output = None

		return predictions_final, vis_output

class AsyncPredictor:
	"""
	A predictor that runs the model asynchronously, possibly on >1 GPUs.
	Because rendering the visualization takes considerably amount of time,
	this helps improve throughput a little bit when rendering videos.
	"""

	class _StopToken:
		pass

	class _PredictWorker(mp.Process):
		def __init__(self, cfg, task_queue, result_queue):
			self.cfg = cfg
			self.task_queue = task_queue
			self.result_queue = result_queue
			super().__init__()

		def run(self):
			predictor = DefaultPredictor(self.cfg)

			while True:
				task = self.task_queue.get()
				if isinstance(task, AsyncPredictor._StopToken):
					break
				idx, data = task
				result = predictor(data)
				self.result_queue.put((idx, result))

	def __init__(self, cfg, num_gpus: int = 1):
		"""
		Args:
			cfg (CfgNode):
			num_gpus (int): if 0, will run on CPU
		"""
		num_workers = max(num_gpus, 1)
		self.task_queue = mp.Queue(maxsize=num_workers * 3)
		self.result_queue = mp.Queue(maxsize=num_workers * 3)
		self.procs = []
		for gpuid in range(max(num_gpus, 1)):
			cfg = cfg.clone()
			cfg.defrost()
			cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
			self.procs.append(
				AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
			)

		self.put_idx = 0
		self.get_idx = 0
		self.result_rank = []
		self.result_data = []

		for p in self.procs:
			p.start()
		atexit.register(self.shutdown)

	def put(self, image):
		self.put_idx += 1
		self.task_queue.put((self.put_idx, image))

	def get(self):
		self.get_idx += 1  # the index needed for this request
		if len(self.result_rank) and self.result_rank[0] == self.get_idx:
			res = self.result_data[0]
			del self.result_data[0], self.result_rank[0]
			return res

		while True:
			# make sure the results are returned in the correct order
			idx, res = self.result_queue.get()
			if idx == self.get_idx:
				return res
			insert = bisect.bisect(self.result_rank, idx)
			self.result_rank.insert(insert, idx)
			self.result_data.insert(insert, res)

	def __len__(self):
		return self.put_idx - self.get_idx

	def __call__(self, image):
		self.put(image)
		return self.get()

	def shutdown(self):
		for _ in self.procs:
			self.task_queue.put(AsyncPredictor._StopToken())

	@property
	def default_buffer_size(self):
		return len(self.procs) * 5
