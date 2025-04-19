"""
    This code is to convert the label type to format, including list, mat, csv, etc.
"""

import os, csv, json

import numpy
from scipy.io import savemat
import numpy as np
from PIL import Image, ImageDraw
from osgeo import gdal

__all__ = ['list2csv', 'json2png', 'txt2mat']


def list2csv(list_file: str, csv_file: str) -> None:

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['images', 'x', 'y', 'labels'])

    with open(list_file, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 1):  # Assuming each image path is followed by its corresponding text file path
            image_path, text_file_path = lines[i].strip().split()  # Split the line to get image and text file paths
            coordinates = read_coordinates_from_text_file(text_file_path)
            image_name = os.path.basename(image_path)
            for coord in coordinates:
                writer.writerow([image_name, coord[0], coord[1], [1]])

def read_coordinates_from_text_file(text_file: str) -> list:
    coordinates = []
    with open(text_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            x, y = map(int, line.strip().split())
            coordinates.append([x, y])
    return coordinates

def png2txt():
    pass

def json2png(json_file: str, tif_file: str, output_png: str) -> None:
    with open(json_file, 'r') as f:
        data = json.load(f)

    dataset = gdal.Open(tif_file)

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    img_size = (height, width)

    img = Image.new('L', img_size, color=0)  # 'L' indicates a grayscale image, with the initial color being 0 (black)
    draw = ImageDraw.Draw(img)

    for shape in data['shapes']:
        points = shape['points']
        points = [(int(x), int(y)) for x, y in points]
        draw.point(points, fill=255, ) # 255 presents white

    # save image
    img.save(output_png)

def txt2mat(target_directory: str, save_path: str) -> None:

    txt_files = [f for f in os.listdir(target_directory) if f.endswith('.txt')]

    for txt_file in txt_files:
        txt_path = os.path.join(target_directory, txt_file)
        mat_data = txt_to_mat(txt_path)

        mat_path = os.path.join(save_path, f"{os.path.splitext(txt_file)[0]}.mat")

        savemat(mat_path, {'data': mat_data})

def txt_to_mat(txt_path: str) -> numpy.array:

    with open(txt_path, 'r') as file:
        lines = file.readlines()

    data = np.array([list(map(float, line.strip().split())) for line in lines])

    return data
