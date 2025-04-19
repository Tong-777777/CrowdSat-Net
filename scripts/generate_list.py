import os

# This script generates the standard '.list' file.
# Note: The directory structure is as follows:
# img/    - Contains image files (e.g., .png)
# txts/   - Contains text files with annotations
# Ensure that the label directory does not contain '.jpg' images to avoid losing point annotations.

def main():
    # train_images = r'D:\PHD_learning\crowd_recognition\datasets\application\merged_data_copy\train\img_aug'
    test_images = r'D:\PHD_learning\crowd_recognition\datasets\google_driver_satellite_update\enhanced\train\img_aug_rename'

    # train_txt_path = r'D:\PHD_learning\crowd_recognition\datasets\application\merged_data_copy\crowd_train.list'
    test_txt_path = r'../data/crowdsat/crowd_train.list'

    image_files = os.listdir(test_images)
    image_files.sort()


    list_file = open(test_txt_path, 'w')

    for train_img in image_files:
        list_file.write(f"{os.path.join(test_images, train_img)} {os.path.join(test_images.replace('img', 'txts'), train_img.replace('.png', '.txt'))}\n")


if __name__ == '__main__':
    main()
