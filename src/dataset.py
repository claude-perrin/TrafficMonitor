import torch
from torch.utils.data import  Dataset
import torchvision.transforms.functional as F
import os
import cv2
import numpy as np
import random

class ObjectDetectionDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.
    Parses XML file
    '''

    def __init__(self, files_dir, transforms=None):
        self.files_dir = files_dir
        self.width, self.height = (640, 480)
        self.transforms = transforms

        self.imgs = [image for image in sorted(os.listdir(files_dir)) if image[-4:] == '.png']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)
        label_filename = img_name[:-4] + '.txt'
        label_file_path = os.path.join(self.files_dir, label_filename)

        # label file
        with open(f"{label_file_path}", 'r') as f:
            txt_annotation = f.readlines()

        labels, target = self._get_image_label(txt_annotation)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img = self._normalize_image(img)
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        #
        if self.transforms:
            # TODO change magic variable
            apply_transformation = random.random() < 0.0
            if apply_transformation:
                img = F.to_pil_image(img)
                img = self.transforms(img)
                if labels[0] != -1:
                    # In a horizontal flip, the x coordinates of the bounding boxes get flipped.
                    # For example, if the original box is [x1, y1, x2, y2], it becomes [width - x2, y1, width - x1, y2]
                    _, width, _ = img.size()
                    target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]

        return img, target

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        return img / 255

    def _get_image_label(self, txt_annotation: list[str]) -> tuple[torch.Tensor, dict]:
        if not txt_annotation:
            labels = torch.tensor([-1])
            target = {
            'area': torch.tensor([0.0]),  
            'boxes': torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            'labels': labels
            }
        else:
            labels = []
            boxes = []
            for annotation in txt_annotation:
                vehicle_type, x, y, width, height = annotation.split()
                labels.append(int(vehicle_type))
                boxes.append([float(x), float(y), float(width), float(height)])

            # convert boxes into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # getting the areas of the boxes
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target = {}
            target["boxes"] = torch.Tensor(boxes)
            target["labels"] = torch.Tensor(labels)
            target["area"] = torch.Tensor(area)
        return labels, target




if __name__ == "__main__":
    # Change common_path
    common_path = "/Users/viktor/polsl/bachelor_project/sagemaker/saved_dataset"
    image_size = (640, 480)
    name2idx = {"cars": 0, "motorcycle": 1}

    train_set = ObjectDetectionDataset(common_path, image_size)

    img, target = train_set[0]
    # cv2.imwrite('image.png', img)
    from dataset_helper import plot_img_bbox
    plot_img_bbox(img, target)

