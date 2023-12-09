import numpy as np
import cv2
import torch
from torchvision.ops import nms

def inference_filter_prediction(outputs, normalized_road_roi_polygon, iou_threshold=0.25, confidence_threshold=0.50):
        cleaned_output = []
        for predicted_dict in outputs:
            mask = predicted_dict['scores'] >= confidence_threshold
            filtered_detections = {k: v[mask] for k, v in predicted_dict.items()}
            filtered_detections = filter_objects_polygon(filtered_detections, normalized_road_roi_polygon)
            if len(filtered_detections['boxes'] != 0):
                print("AFTER CLEANING ON THE ROAD", filtered_detections)
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
        print("Filtered based on  polygon: ", filtered_detections)
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

def normalize_polygon(image_dim, ROAD_ROI_POLYGON):
    image_width, image_height = image_dim
    normalized_polygon = [(int(point[0] * image_width), int(point[1] * image_height)) for point in ROAD_ROI_POLYGON]
    return normalized_polygon