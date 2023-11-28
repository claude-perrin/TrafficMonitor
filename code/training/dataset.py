import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
import os
import cv2
import random
from conf import *


class ObjectDetectionDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    '''

    def __init__(self, files_dir, transforms=None):
        self.files_dir = files_dir
        self.transforms = transforms
        self.allowed_image_formats = ["png", "jpeg", "jpg"]
        self.imgs = self.get_images()
        self.image_width, self.image_height = self.get_image_size()
        self.tranformation_probability = 0.7



    def __len__(self):
        return len(self.imgs)
    
    
    def get_images(self):
        images = []
        for image in sorted(os.listdir(self.files_dir)):
            if self._get_image_format(image) in self.allowed_image_formats:
                label_filename = self._get_image_name(image) + '.txt'
                if not os.path.exists(f"{self.files_dir}/{label_filename}"):
                    print(f"FOUND INCONSISTENT FILE: {image}, \nlabel not found: {label_filename}")
                else:
                    images.append(image)
        return images
    
    def get_image_size(self):
        img_name = self.imgs[0]
        image_path = os.path.join(self.files_dir, img_name)
        img = cv2.imread(image_path).shape
        
        return (img[0], img[1])

    
    def _get_image_name(self, image_name):
        return image_name.rsplit('.', 1)[0]

    def _get_image_format(self, image_name):
        return image_name.rsplit('.', 1)[-1]

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)
        label_filename = self._get_image_name(img_name) + '.txt'

        target = self._get_target(label_filename)
        grayscale, original_image = self._get_images(image_path)

        if self.transforms:
            grayscale, target = self._apply_transform(grayscale, target)
        idx = torch.tensor(idx)
        return (grayscale, target, original_image, idx)

    def _get_target(self, label_filename):
        label_file_path = os.path.join(self.files_dir, label_filename)
        with open(f"{label_file_path}", 'r') as f:
            txt_annotation = f.readlines()
        if not txt_annotation:
            boxes, labels, area = self.get_empty_target()
        else:
            labels = []
            boxes = []
            for annotation in txt_annotation:
                vehicle_type, x, y, width, height = annotation.split()
                labels.append(int(vehicle_type))
                x_min, y_min, x_max, y_max = self.denormalize_box(float(x), float(y), float(width), float(height))
                # print(boxes)
                # print(f"{x_min}, {y_min}, {x_max}, {y_max}")
                boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
            boxes, labels, area = self._format_targers(boxes, labels)
            
        return (boxes, labels, area)
    

    def denormalize_box(self, x_min, y_min, width, height):
        # x_min *= self.width
        # y_min *= self.height
        # width *= self.width
        # height *= self.height
        # x_max = x_min + width
        # y_max = y_min + height
        x = float(x_min) * self.image_width
        y = float(y_min) * self.image_height
        width = float(width) * self.image_width
        height = float(height) * self.image_height

        top_left_x = x - width / 2
        top_left_y = y - height / 2
        bottom_right_x = x + width / 2
        bottom_right_y = y + height / 2
        return [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        # return [x_min, y_min, x_max, y_max]

    def _format_targers(self, boxes, labels):
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return boxes, labels, area
    
    def _get_images(self, image_path):
        # reading the images and converting them to correct size and color
        original_image = cv2.imread(image_path)
        grayscale = self._to_grayscale(original_image)
        grayscale = self._normalize_image(grayscale)
        grayscale = torch.from_numpy(grayscale).float()
        grayscale = grayscale.unsqueeze(0)
        return grayscale, torch.from_numpy(original_image)

    def _to_grayscale(self, image):	
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def _normalize_image(self, img):
        return img / 255

    def _apply_transform(self, img, target):
        boxes, labels, _ = target
        # TODO change magic variable
        apply_transformation = random.random() < self.tranformation_probability
        if apply_transformation:
            img = F.to_pil_image(img)
            img = self.transforms(img)
            if labels[0] != -1:
                # In a horizontal flip, the x coordinates of the bounding boxes get flipped.
                # For example, if the original box is [x1, y1, x2, y2], it becomes [width - x2, y1, width - x1, y2]
                _, width, _ = img.size()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return img, target

    @staticmethod
    def get_empty_target():
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        labels = torch.tensor([CLASSES_TO_IDX["background"]], dtype=torch.int64)
        area = torch.tensor([1.0])
        return boxes, labels, area


if __name__ == "__main__":
    # Change common_path
    common_path = "/Users/viktor/polsl/bachelor_project/sagemaker/test_set"
    image_size = (640, 480)
    # name2idx = {"cars": 0, "motorcycle": 1}

    train_set = ObjectDetectionDataset(common_path, image_size)
    train_set[0]
    # dl = DataLoader(train_set, batch_size=1)
    # print(dl[0])
    # for i,z,x in dl:
    #     print(i)
    # from dataset_helper import plot_img_bbox

    # plot_img_bbox(img, target)
