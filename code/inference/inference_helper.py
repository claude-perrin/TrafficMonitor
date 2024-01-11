import numpy as np
import cv2
import torch
from torchvision.ops import nms
from conf import *

def inference_filter_prediction(outputs, roi_polygon=None, iou_threshold=0.25, confidence_threshold=0.50):
        cleaned_output = []
        for predicted_dict in outputs:
            mask = predicted_dict['scores'] >= confidence_threshold
            filtered_detections = {k: v[mask] for k, v in predicted_dict.items()}
            if roi_polygon != None:
                filtered_detections = filter_objects_polygon(filtered_detections, roi_polygon)
            if len(filtered_detections['boxes'] != 0):
                nms_indices = nms(
                    filtered_detections['boxes'],
                    filtered_detections['scores'],
                    iou_threshold
                )
                filtered_detections = {k: v[nms_indices] for k, v in filtered_detections.items()}
            cleaned_output.append(filtered_detections)
        return cleaned_output

def filter_objects_polygon(detections, normalized_road_roi_polygon):
        boxes = []
        filtered_detections = {}
        for box in detections['boxes']:
            # Check if any corner of the box is inside the road ROI
            box_corners = np.array([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
            if any(point_in_polygon(corner, normalized_road_roi_polygon) for corner in box_corners):
                boxes.append(box)
        if len(boxes) != 0:
            filtered_detections['scores'] = detections['scores'][:len(boxes)].clone().detach()
            filtered_detections['labels'] = detections['labels'][:len(boxes)].clone().detach()
            filtered_detections['boxes'] = torch.stack(boxes)
        else:
            filtered_detections = default_return_output()
        return filtered_detections

def default_return_output():
    print("RETURNING DEFAULT DICT")
    return {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([], dtype=torch.int64)}

def point_in_polygon(point, polygon):
        poly_mask = np.array(polygon, dtype=np.int32)
        poly_mask = poly_mask.reshape((-1, 1, 2))
        return cv2.pointPolygonTest(poly_mask, point, False) >= 0

def preprocess_image(img):
        grayscale = to_grayscale(img)
        grayscale = normalize_image(grayscale)
        grayscale = torch.from_numpy(grayscale).float()
        grayscale = grayscale.unsqueeze(0)
        return grayscale

def to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale

def normalize_image(img):
    return img / 255

def denormalize_polygon(image_dim, ROAD_ROI_POLYGON):
    image_width, image_height = image_dim
    normalized_polygon = [(int(point[0] * image_width), int(point[1] * image_height)) for point in ROAD_ROI_POLYGON]
    return normalized_polygon


def plot_boxes(results, frame, denormalized_road_roi_polygon=None):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    label_bg_white = (255, 255, 255)
    if len(results) != 0:
        result = results
        # for result in results:
        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
            label = label.item()
            score = score.item()
            box_color = BOX_COLOR[label]
            box_color = (128, 0, 128)
            # box_color = (255,255,255)
            x1, x2, x3, x4 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
            cv2.rectangle(frame, (x1,x2),(x3,x4), box_color, 2)
            cv2.rectangle(frame, (x1, x2-25), (x1+150, x2), label_bg_white, -1)
            label_text = f'{class_to_label(label)}: {score:.2f}'
            cv2.putText(frame, label_text, (x1, x2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
    cv2.polylines(frame, [np.array(denormalized_road_roi_polygon)], isClosed=True, color=(32, 32, 128), thickness=2)
    return frame

def class_to_label(label):
    return IDX_TO_CLASSES[label]
