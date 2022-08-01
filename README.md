# What's in the Black Box? The False Negative Mechanisms Inside Object Detectors

This is the official repository of paper:

**[What's in the Black Box? The False Negative Mechanisms Inside Object Detectors](https://arxiv.org/abs/2203.00807)**

*Dimity Miller, Peyman Moghadam, Mark Cox, Matt Wildie, Raja Jurdak*

<!-- ![FrontPage](images/IROSRALFrontPage.jpg) -->
<p align="center">
  <img width="600" height="281" src=images/IROSRALFrontPage.jpg>
</p>

If you use this repository, please cite:

```text
@article{miller2022s,
  title={What's in the Black Box? The False Negative Mechanisms Inside Object Detectors},
  author={Miller, Dimity and Moghadam, Peyman and Cox, Mark and Wildie, Matt and Jurdak, Raja},
  journal={IEEE Robotics and Automation Letters}, 
  year={2022},
  volume={7},
  number={3},
  pages={8510-8517},
  doi={10.1109/LRA.2022.3187831}
}
```

**Contact**

If you have any questions or comments, please contact [Dimity Miller](mailto:d24.miller@qut.edu.au).

## Installation

This code was developed with Python 3.8 on Ubuntu 20.04. 

### Clone and Install Detectron2 Repository
1. Clone [detectron2](https://github.com/facebookresearch/detectron2) inside the fn_mechanisms folder.
```bash
cd fn_mechanisms
git clone https://github.com/facebookresearch/detectron2.git
```
2. Follow the [detectron2 instructions for installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). We use pytorch 1.12 with cuda 11.6, and build detectron2 from source. However, you should be able to use other versions of pytorch and cuda as long as they meet the listed detectron2 requirements.
3. You should be able to run the following command with no errors. If you have any errors, this is an issue with your detectron2 installation and you should debug or raise an issue with the detectron2 repository.
```bash
cd detectron2/demo
python demo.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --input ../../images/test_im.jpg --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl
```
## Data Setup
**Quick Setup:**

COCO data can be downloaded from [here](https://cocodataset.org/#download). The following commands can be used to quickly download the COCO val2017 images and annotations for evaluating. 
```bash
mkdir data
mkdir coco
cd coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

```
**Other Datasets:**

The scripts are designed to be input folders that contain the images to be tested (and no other file types), and an annotation file in the COCO Object Detection format. Read [here](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format) for details on how to format the annotation file. The annotation file does not need segmentations or segmentation-related information, but all other fields are necessary.


## Pre-trained Models
We use pre-trained models (trained on COCO) from the [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md). 

## How to use
### Identifying False Negative Mechanisms for Detectron2-based Detectors
```bash
python identify_fn.py --input image_folder --gt annotation_file --opts MODEL.WEIGHTS weights_file
```
where:
* `image_folder` is the path to the folder containing all images to be tested. No other files should be in this folder.
* `annotation_file` is the path to the json file containing annotations for all images, in the COCO Object Detection format (see above).
* `weights_file` is the path to the weights file to test the 

Optional arguments:

* `--config-file config_path` where config_path is a string of the path to the detectron2 detector config file. Default is "detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml".
* `--confidence-threshold c` where c ris a float of the minimum class confidence score for detections to be considered valid. Default is 0.3.
* `--vis True` to visualise each the detections for each image
* `--visFN True` to visualise each image's false negative objects and a visualisation of the false negative mechanism responsible.
* `--opts` can be also be used to alter the config file options. See detectron2 instructions for more information. 

**Testing Faster R-CNN (R50 FPN 3x) on COCO**

After following the instructions above for downloading the COCO val2017 data:
```bash
python identify_fn.py --input data/coco/val2017/ --gt data/coco/annotations/instances_val2017.json --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl
```
**Testing RetinaNet (R50 FPN 3x) on COCO**

After following the instructions above for downloading the COCO val2017 data:
```bash
python identify_fn.py --input data/coco/val2017/ --gt data/coco/annotations/instances_val2017.json --config-file detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml --opts MODEL.WEIGHTS https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl
```
**Results**

Running the two commands above should generate the following results:
<table>
  <tr>
    <td colspan = '2'></td>
    <td align = 'center' colspan = '5'>False Negative Mechanisms</td>
  </tr>
  <tr>
    <td align = 'center'>Detector</td>
    <td align = 'center'># of False Negatives</td>
    <td align = 'center'>Proposal Process</td>
    <td align = 'center'>Regressor</td>
    <td align = 'center'>Interclass Classification</td>
    <td align = 'center'>Background Classification</td>
    <td align = 'center'>Classifier Calibration</td>
  </tr>
  <tr>
    <td align = 'center'>Faster R-CNN (R50 FPN 3x)</td>
    <td align = 'center'>10464</td>
    <td align = 'center'>20.01%</td>
    <td align = 'center'>2.48%</td>
    <td align = 'center'>12.19%</td>
    <td align = 'center'>58.22%</td>
    <td align = 'center'>7.10%</td>
  </tr>
    <tr>
    <td align = 'center'>RetinaNet (R50 FPN 3x)</td>
    <td align = 'center'>11869</td>
    <td align = 'center'>5.87%</td>
    <td align = 'center'>0.07%</td>
    <td align = 'center'>9.52%</td>
    <td align = 'center'>77.86%</td>
    <td align = 'center'>6.68%</td>
  </tr>
</table>

### Visualising False Negative Mechanisms

When running the identify_fn.py script, you can set --visFN True to visualise false negative mechanisms. This section explains what is being visualised with some examples.

**Proposal Process Mechanism**

**Regressor Mechanism**

**Interclass Classification Mechanism**

**Background Classification Mechanism**

**Classifier Calibration Mechanism**


## Acknowledgement
This code builds upon the [detectron2 repository](https://github.com/facebookresearch/detectron2). Please also acknowledge detectron2 if you use this repository.
