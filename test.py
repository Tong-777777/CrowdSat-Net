import csv
import re
import cv2
import torch.nn
from datasets.transforms import *
import albumentations as A
from datasets.crowd_dataset import CrowdDataset
import warnings
import os
import time
import argparse
import datetime
import yaml
import pprint
import pandas as pd
from utils.loss import FocalLoss
from utils.seed import set_seed
from torch import Tensor
from torch.nn import CrossEntropyLoss
from model import LossWrapper, CrowdSatNet, load_model
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from utils.logger import *
from utils.averager import *
from utils.metrics import *
from typing import Optional, Union
from utils.lmds import LMDS
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for Testing CrowdSat-Net', add_help=False)
    parser.add_argument("--config", default="./configs/crowdsat.yaml", help="Path to the config file (yaml type)")

    return parser.parse_args()


def main(args):
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    ###########################
    # 1. seed and output settings
    ###########################
    set_seed(cfg["seed"])

    device = torch.device(cfg["device_name"])

    work_dir = cfg["work_dir"]
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    logger_path = os.path.join(cfg["work_dir"], cfg["output"])
    os.makedirs(logger_path, exist_ok=True)
    prediction_visual = os.path.join(logger_path, 'prediction_visual')
    os.makedirs(prediction_visual, exist_ok=True)
    evaluation_visual = prediction_visual.replace('prediction_visual', 'evaluation_visual')
    os.makedirs(evaluation_visual, exist_ok=True)

    logger, curr_timestr = setup_default_logging("global", logger_path)
    logger.info("{}".format(pprint.pformat(cfg)))

    csv_path = os.path.join(logger_path, "each_image_information.csv")
    file = open(csv_path, 'w', newline="")

    writer = csv.writer(file)
    writer.writerow(['ID', 'gt', 'P', 'R', 'F1', 'TP', 'FP', 'FN'])

    ###########################
    # 2. model preparation
    ###########################
    model = CrowdSatNet(num_classes=cfg["datasets"]["num_classes"])
    model.to(device)

    model = LossWrapper(model, mode='preds_only')

    ###########################
    # 3. datasets preparation
    ###########################
    test_set = CrowdDataset(
        data_root=cfg["train"]["data_root"],
        train=False,
        albu_transforms=[A.Normalize(p=cfg["train"]["validate"]["albu_transforms"]["Normalize"]["p"])],
        end_transforms=[DownSample(down_ratio=cfg["train"]["validate"]["end_transforms"]["DownSample"]["down_ratio"],
                                   crowd_type=cfg["train"]["validate"]["end_transforms"]["DownSample"]["crowd_type"])]
    )

    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=cfg["val_settings"]["batch_size"],
        shuffle=False
    )

    ###########################
    # 4. testing process
    ###########################

    metrics = PointsMetrics(radius=2, num_classes=cfg['datasets']['num_classes'])

    checkpoint = cfg["val_settings"]["checkpoint"]
    checkpoint_path = os.path.join(work_dir, checkpoint+'_model.pth')
    model = load_model(model, checkpoint_path, strict=False)

    metrics.flush()
    iter_metrics = metrics.copy()


    logger.info('-------------------------- start testing --------------------------')
    model.eval()

    for step, (images, targets) in enumerate(test_dataloader):

        image_index = re.findall(r'\d+', targets['img_path'][0])
        images = images.cuda()
        output = model(images)

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()

        gt = dict(
            loc=gt_coords,
            labels=gt_labels
        )

        lmds = LMDS(kernel_size=cfg["val_settings"]["lmds_kwargs"]["kernel_size"],
                    adapt_ts=cfg["val_settings"]["lmds_kwargs"]["adapt_ts"])

        counts, locs, labels, scores = lmds(output)

        preds = dict(
            loc=locs[0],
            labels=labels[0],
            scores=scores[0],
        )

        iter_metrics.feed(**dict(gt=gt, preds=preds))

        f1 = round(iter_metrics.fbeta_score(), 5)
        precision = round(iter_metrics.precision(), 5)
        recall = round(iter_metrics.recall(), 5)

        tp_points = iter_metrics.current_tp
        fp_points = iter_metrics.current_fp
        fn_points = iter_metrics.current_fn

        writer.writerow([image_index, len(gt_coords), precision, recall, f1, len(tp_points), len(fp_points), len(fn_points)])

        iter_metrics.aggregate()

        iter_metrics.flush()
        metrics.feed(**dict(gt=gt, preds=preds))

        # save prediction results
        img_raw = Image.open(targets['img_path'][0]).convert('RGB')
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        img_to_draw_new = img_to_draw.copy()
        for pi in locs[0]:
            img_to_draw = cv2.circle(img_to_draw, (int(pi[1])*2, int(pi[0])*2), 2, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(prediction_visual, image_index[0] + '_pred{}.png'.format(len(locs[0]))),
                    img_to_draw)

        # save evaluation results
        scale_factor = 2

        draw_configs = [
            (tp_points, cv2.circle, {
                "radius": 4,
                "color": (255, 255, 0),
                "thickness": -1
            }),
            (fp_points, cv2.circle, {
                "radius": 4,
                "color": (255, 0, 255),
                "thickness": 2
            }),
            (fn_points, cv2.drawMarker, {
                "color": (0, 255, 255),
                "markerType": cv2.MARKER_CROSS,
                "markerSize": 8,
                "thickness": 2
            })
        ]

        for points, draw_func, params in draw_configs:
            for point in points:

                scaled_x = int(point[1] * scale_factor)
                scaled_y = int(point[0] * scale_factor)


                draw_func(
                    img_to_draw_new,
                    (scaled_x, scaled_y),
                    ** params
                )


        cv2.imwrite(os.path.join(evaluation_visual, image_index[0] + f'_eva.png'), img_to_draw_new)


    metrics.aggregate()

    recall = metrics.recall()
    precision = metrics.precision()
    f1_score = metrics.fbeta_score()

    tmp_results = {
        'f1_score': f1_score,
        'recall': recall,
        'precision': precision,
    }

    print(tmp_results)


if __name__ == '__main__':
    args = get_args_parser()

    main(args)

