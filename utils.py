import numpy as np

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
