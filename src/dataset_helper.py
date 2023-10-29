import numpy as np
import torch
import os
import xml.etree.ElementTree as ET

from matplotlib import patches, pyplot as plt
from torch import ops



def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()


def remove_empty_images(files_dir):
    imgs = [image for image in sorted(os.listdir(files_dir)) if image[-4:] == '.png']

    for img_name in imgs:
        image_path = os.path.join(files_dir, img_name)
        label_filename = img_name[:-4] + '.txt'
        label_file_path = os.path.join(files_dir, label_filename)

        with open(f"{label_file_path}", 'r') as f:
            txt_annotation = f.readlines()

        if txt_annotation:
            pass
        else:
            print("remove", image_path)
            os.remove(image_path)
            os.remove(label_file_path)



if __name__ == "__main__":
    files_dir = "/Users/viktor/polsl/bachelor_project/RcnnTraining/dataset/train_set"
    remove_empty_images(files_dir=files_dir)
