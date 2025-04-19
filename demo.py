import argparse
from PIL import Image
import albumentations as A
import numpy as np
import os
import sys
from model import LossWrapper, CrowdSatNet, load_model
import torch
from utils.lmds import LMDS
import cv2
import torchvision

def get_script_dir():

    return os.path.dirname(os.path.abspath(sys.argv[0]))

script_dir = get_script_dir()

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for demo', add_help=False)

    default_img = os.path.join(script_dir, 'demo', 'red_square.png')
    default_output = os.path.join(script_dir, 'demo', 'red_square_prediction.png')

    parser.add_argument("--img_path", default=default_img)
    parser.add_argument("--output_path", default=default_output)
    parser.add_argument("--checkpoint_path", default=r'D:\PHD_learning\crowd_recognition\datasets\google_driver_satellite_update\work\best_model.pth')

    return parser.parse_args()

def demo(args):

    # data
    img_path = args.img_path
    output_path = args.output_path
    checkpoint_path = args.checkpoint_path

    img = Image.open(img_path).convert('RGB')
    numpy_image = np.array(img)

    transform = A.Compose([
        A.Normalize(
            p=1.0
        )
    ])

    transformed = transform(image=numpy_image)

    _img = transformed["image"]
    _img = torchvision.transforms.ToTensor()(_img)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _img = _img.to(device)
    _img = _img.unsqueeze(0)

    # model
    model = CrowdSatNet(num_classes=2)
    model.to(device)

    model = LossWrapper(model, mode='preds_only')
    model = load_model(model, checkpoint_path, strict=False)

    pre = model(_img)

    lmds = LMDS(kernel_size=(3, 3),
                adapt_ts=0.1)

    counts, locs, labels, scores = lmds(pre)

    img_to_draw = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    for pi in locs[0]:
        img_to_draw = cv2.circle(img_to_draw, (int(pi[1]) * 2, int(pi[0]) * 2), 2, (0, 0, 255), -1)

    cv2.imwrite(output_path, img_to_draw)

if __name__ == '__main__':
    args = get_args_parser()

    demo(args=args)
