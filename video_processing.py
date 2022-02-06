# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import json
import requests
import urllib.request

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog

# Build the detector and download our pretrained weights
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
predictor = DefaultPredictor(cfg)

# constants
WINDOW_NAME = "Detic"
from detic.modeling.utils import reset_cls_test
# Setup the model's vocabulary using build-in datasets

BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)

def setup_cfg():
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file('configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
    cfg.merge_from_list('MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth')
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.6
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    # if not args.pred_all_class:
    #     cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg




class task_Worker():

    def run(self, task):
        if task.get('storage') is None:
            task['storage'] = '/storage'
        }
        mp.set_start_method("spawn", force=True)
        # task = get_parser().parse_task()
        setup_logger(name="fvcore")
        logger = setup_logger()
        # logger.info("Arguments: " + str(task))
        try:
            os.rmdir(task['storage'] + '/frames')
            os.remove(task['storage'] +'/temp.mp4')
        except OSError:
            pass
        try:
            os.makedirs(task['storage'] + '/frames')
        except OSError:
            pass


        urllib.request.urlretrieve(task['url'], task['storage'] + '/temp.mp4')
        os.system('ffmpeg -i ' + task['storage'] +'/temp.mp4 -vf "select=eq(pict_type\,I)" -vsync vfr ' + task['storage'] + '/frames/frame-%02d.png')

        frames = []
        result = []
        for img in sorted(glob.glob(task['storage'] + "/frames/*.png")):
            cv_img = cv2.imread(img)
            frames.append(cv_img)

        # print(len(frames))
        for img in frames:
          outputs = predictor(img)
          result.append(outputs)


        final_response = []
        categories = [metadata.thing_classes[x] for x in result[0]["instances"].pred_classes.cpu().tolist()] # class names
        for i in range(len(result)):
          frame_no = i
          frame_data = {
            "frame_no" : frame_no,
            "annotations" : []
          }
          categories = [metadata.thing_classes[x] for x in result[i]["instances"].pred_classes.cpu().tolist()] # class names

          for j in range(len(categories)):
            score = float(result[i]["instances"].scores[j])

            boxes = []
            for el in result[i]["instances"].pred_boxes:
              tup = (int(el[0]), int(el[1]), int(el[2]), int(el[3]))
              boxes.append(el)

            x1 = int(boxes[j][0])
            y1 = int(boxes[j][1])
            x2 = int(boxes[j][2])
            y2 = int(boxes[j][3])
            left = x1
            top = y1
            width = x2 - x1
            height = y2 - y2
            item = {
                "object-score" : score,
                "object-name" : categories[j],
                "left" : left,
                "top" : top,
                "width" : width,
                "height" : height
            }
            frame_data["annotations"].append(item)


          final_response.append(frame_data)
        # print(final_response)
        json_string = json.dumps(final_response)
        with open(task['storage'] + '/' + task['id'] + '.json', 'w') as outfile:
            outfile.write(json_string)
