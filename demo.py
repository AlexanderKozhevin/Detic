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

def setup_cfg(args):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file('configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
    cfg.merge_from_list('MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth')
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--id",
        type=str,
        help="path to config file",
    )
    parser.add_argument(
        '--keyframes_only',
        type=int,
        help='url',
        default=1
    )
    parser.add_argument(
        '--storage', type=str, help='storage', default="/storage"
    ),
    parser.add_argument(
        '--url', type=str, help='url'
    )
    # parser.add_argument(
    #     "--config-file",
    #     default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
    #     metavar="FILE",
    #     help="path to config file",
    # )
    # parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    # parser.add_argument("--video-input", help="Path to video file.")
    # parser.add_argument(
    #     "--input",
    #     nargs="+",
    #     help="A list of space separated input images; "
    #     "or a single glob pattern such as 'directory/*.jpg'",
    # )
    # parser.add_argument(
    #     "--output",
    #     help="A file or directory to save output visualizations. "
    #     "If not given, will show output in an OpenCV window.",
    # )
    # parser.add_argument(
    #     "--vocabulary",
    #     default="lvis",
    #     choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
    #     help="",
    # )
    # parser.add_argument(
    #     "--custom_vocabulary",
    #     default="",
    #     help="",
    # )
    # parser.add_argument("--pred_all_class", action='store_true')
    # parser.add_argument(
    #     "--confidence-threshold",
    #     type=float,
    #     default=0.5,
    #     help="Minimum score for instance predictions to be shown",
    # )
    # parser.add_argument(
    #     "--opts",
    #     help="Modify config options using the command-line 'KEY VALUE' pairs",
    #     default=[],
    #     nargs=argparse.REMAINDER,
    # )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    try:
        os.rmdir(args.storage + '/frames')
    except OSError:
        pass
    try:
        os.makedirs(args.storage + '/frames')
    except OSError:
        pass


    urllib.request.urlretrieve(args.url, args.storage + '/temp.mp4')
    os.system('ffmpeg -i ' + args.storage +'/temp.mp4 -vf "select=eq(pict_type\,I)" -vsync vfr ' + args.storage + '/frames/frame-%02d.png')

    frames = []
    result = []
    for img in sorted(glob.glob(args.storage + "/frames/*.png")):
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
    with open(args.storage + '/' + args.id + '.json', 'w') as outfile:
        outfile.write(json_string)

    # im = cv2.imread("/content/desk.jpg")
    #
    # outputs = predictor(im)
    # print(outputs)

    # cfg = setup_cfg(args)

    # demo = VisualizationDemo(cfg, args)

    # if args.input:
    #     if len(args.input) == 1:
    #         args.input = glob.glob(os.path.expanduser(args.input[0]))
    #         assert args.input, "The input path(s) was not found"
    #     for path in tqdm.tqdm(args.input, disable=not args.output):
    #         img = read_image(path, format="BGR")
    #         start_time = time.time()
    #         predictions, visualized_output = demo.run_on_image(img)
    #         logger.info(
    #             "{}: {} in {:.2f}s".format(
    #                 path,
    #                 "detected {} instances".format(len(predictions["instances"]))
    #                 if "instances" in predictions
    #                 else "finished",
    #                 time.time() - start_time,
    #             )
    #         )
    #
    #         if args.output:
    #             if os.path.isdir(args.output):
    #                 assert os.path.isdir(args.output), args.output
    #                 out_filename = os.path.join(args.output, os.path.basename(path))
    #             else:
    #                 assert len(args.input) == 1, "Please specify a directory with args.output"
    #                 out_filename = args.output
    #             visualized_output.save(out_filename)
    #         else:
    #             cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #             cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    #             if cv2.waitKey(0) == 27:
    #                 break  # esc to quit
    # elif args.webcam:
    #     assert args.input is None, "Cannot have both --input and --webcam!"
    #     assert args.output is None, "output not yet supported with --webcam!"
    #     cam = cv2.VideoCapture(0)
    #     for vis in tqdm.tqdm(demo.run_on_video(cam)):
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #         cv2.imshow(WINDOW_NAME, vis)
    #         if cv2.waitKey(1) == 27:
    #             break  # esc to quit
    #     cam.release()
    #     cv2.destroyAllWindows()
    # elif args.video_input:
    #     video = cv2.VideoCapture(args.video_input)
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frames_per_second = video.get(cv2.CAP_PROP_FPS)
    #     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     basename = os.path.basename(args.video_input)
    #     codec, file_ext = (
    #         ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    #     )
    #     if codec == ".mp4v":
    #         warnings.warn("x264 codec not available, switching to mp4v")
    #     if args.output:
    #         if os.path.isdir(args.output):
    #             output_fname = os.path.join(args.output, basename)
    #             output_fname = os.path.splitext(output_fname)[0] + file_ext
    #         else:
    #             output_fname = args.output
    #         assert not os.path.isfile(output_fname), output_fname
    #         output_file = cv2.VideoWriter(
    #             filename=output_fname,
    #             # some installation of opencv may not support x264 (due to its license),
    #             # you can try other format (e.g. MPEG)
    #             fourcc=cv2.VideoWriter_fourcc(*codec),
    #             fps=float(frames_per_second),
    #             frameSize=(width, height),
    #             isColor=True,
    #         )
    #     assert os.path.isfile(args.video_input)
    #     for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
    #         if args.output:
    #             output_file.write(vis_frame)
    #         else:
    #             cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
    #             cv2.imshow(basename, vis_frame)
    #             if cv2.waitKey(1) == 27:
    #                 break  # esc to quit
    #     video.release()
    #     if args.output:
    #         output_file.release()
    #     else:
    #         cv2.destroyAllWindows()
