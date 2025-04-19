import os
import cv2
import numpy as np
from random import randint, choice

# This script performs data augmentation, including horizontal and vertical flipping, as well as CutMix.
# Notes: Both img_dir and label_dir should be directories containing images in '.png' format.
# Note: The label_dir should not contain '.jpg' images, as they may lose point annotations.

img_dir = r"D:\PHD_learning\crowd_recognition\datasets\application\merged_data_copy\train\img"
label_dir = r"D:\PHD_learning\crowd_recognition\datasets\application\merged_data_copy\train\label"
output_img_dir = r"D:\PHD_learning\crowd_recognition\datasets\application\merged_data_copy\train\img_aug"
output_label_dir = r"D:\PHD_learning\crowd_recognition\datasets\application\merged_data_copy\train\label_aug"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# CutMix
def cutmix(image1, label1, image2, label2):
    h, w, _ = image1.shape
    cut_x = randint(0, w // 2)
    cut_y = randint(0, h // 2)
    cut_w = randint(w // 4, w // 2)
    cut_h = randint(h // 4, h // 2)

    x1, x2 = cut_x, min(cut_x + cut_w, w)
    y1, y2 = cut_y, min(cut_y + cut_h, h)

    mixed_image = image2.copy()
    mixed_label = label2.copy()

    mixed_image[y1:y2, x1:x2] = image1[y1:y2, x1:x2]
    mixed_label[y1:y2, x1:x2] = label1[y1:y2, x1:x2]

    return mixed_image, mixed_label

img_names = [name for name in os.listdir(img_dir) if name.endswith(".png")]

for img_name in img_names:
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name)

    image = cv2.imread(img_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # h_flip
    h_flip_img = cv2.flip(image, 1)
    h_flip_label = cv2.flip(label, 1)

    cv2.imwrite(os.path.join(output_img_dir, f"hflip_{img_name}"), h_flip_img)
    cv2.imwrite(os.path.join(output_label_dir, f"hflip_{img_name}"), h_flip_label)

    # v_flip
    v_flip_img = cv2.flip(image, 0)
    v_flip_label = cv2.flip(label, 0)
    cv2.imwrite(os.path.join(output_img_dir, f"vflip_{img_name}"), v_flip_img)
    cv2.imwrite(os.path.join(output_label_dir, f"vflip_{img_name}"), v_flip_label)

    # CutMix
    random_img_name = choice(img_names)
    random_img_path = os.path.join(img_dir, random_img_name)
    random_label_path = os.path.join(label_dir, random_img_name)

    random_image = cv2.imread(random_img_path)
    random_label = cv2.imread(random_label_path, cv2.IMREAD_GRAYSCALE)

    cutmix_img, cutmix_label = cutmix(image, label, random_image, random_label)
    cv2.imwrite(os.path.join(output_img_dir, f"cutmix_{img_name}"), cutmix_img)
    cv2.imwrite(os.path.join(output_label_dir, f"cutmix_{img_name}"), cutmix_label)
