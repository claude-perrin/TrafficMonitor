import cv2
import torchvision
import numpy as np
import torch
import urllib
from conf import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from process_artifacts import process_inference_output
import os
from flask import Flask, Response


model_path = "epoch-4_model.pth"
app = Flask(__name__)

class ObjectDetection:
    """
    Class implements FasterRcnn model to make inferences.
    """

    def __init__(self, normalized_road_roi_polygon):
        self.model = self.load_model()
        self.classes = list(CLASSES_TO_IDX.keys())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.DOCKER_RUN = int(os.getenv("DOCKER_RUNNING"))
        if self.DOCKER_RUN == "1":
            print("RUNNING INSIDE OF DOCKER")
        else:
            print("RUNNING LOCALLY")
        self.normalized_road_roi_polygon = normalized_road_roi_polygon



    def load_model(self, num_classes=NUMBER_OF_CLASSES):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        state_dict= torch.load(model_path)
        updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
        model.load_state_dict(updated_state)
        return model

    def __call__(self, img):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        """
        grayscale = self.__preprocess_image(img)
        output = self.__inference(self.model, grayscale)
        output = self.__inference_filter_prediction(output)
        return output


    def __preprocess_image(self, img):
        grayscale = self.__to_grayscale(img)
        grayscale = self.__normalize_image(grayscale)
        grayscale = torch.from_numpy(grayscale).float()
        grayscale = grayscale.unsqueeze(0)
        return grayscale

    def __to_grayscale(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale

    def __normalize_image(self, img):
        return img / 255

    def __inference(self, model, input_data):
        input_data = input_data.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            output = model(input_data)
        print("Inference output: ", output)
        return output

    def __inference_filter_prediction(self, outputs, iou_threshold=0.25, confidence_threshold=0.50):
        cleaned_output = []
        for predicted_dict in outputs:
            mask = predicted_dict['scores'] >= confidence_threshold
            filtered_detections = {k: v[mask] for k, v in predicted_dict.items()}
            filtered_detections = self.filter_objects_polygon(filtered_detections)
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

    def filter_objects_polygon(self, detections):
        boxes = []
        filtered_detections = {}
        for box in detections['boxes']:
            # Check if any corner of the box is inside the road ROI
            box_corners = np.array([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
            if any(self.point_in_polygon(corner, self.normalized_road_roi_polygon) for corner in box_corners):
                boxes.append(box)
        if len(boxes) != 0:
            filtered_detections['scores'] = detections['scores'][:len(boxes)].clone().detach()
            filtered_detections['labels'] = detections['labels'][:len(boxes)].clone().detach()
            filtered_detections['boxes'] = torch.stack(boxes)
        else:
            filtered_detections = self.default_return_output()
        print("Filtered based on  polygon: ", filtered_detections)
        return filtered_detections

    def point_in_polygon(self, point, polygon):
        poly_mask = np.array(polygon, dtype=np.int32)
        poly_mask = poly_mask.reshape((-1, 1, 2))
        return cv2.pointPolygonTest(poly_mask, point, False) >= 0


    def default_return_output(self):
        print("RETURNING DEFAULT DICT")
        return {'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([], dtype=torch.int64)}

    def __class_to_label(self, x):
        return IDX_TO_CLASSES[int(x)]



@app.route('/video')
def video_feed():
    return Response(start_loop(),
                mimetype='multipart/x-mixed-replace; boundary=frame')

def start_loop():
    image_dim = (600,800)
    normalized_road_roi_polygon = normalize_polygon(image_dim, ROAD_ROI_POLYGON)
    a = ObjectDetection(normalized_road_roi_polygon)

    while True:
        img = get_image(camera_ip)
        output = a(img)
        print("OUTPUT: ", output)
        frame = plot_boxes(normalized_road_roi_polygon, output, img)
        process_inference_output(output, camera_ip)
        yield stream_to_localhost(frame)


def get_image(camera_ip):
    imgResp=urllib.request.urlopen(camera_ip)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    return img


def plot_boxes(normalized_road_roi_polygon, results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    if len(results) != 0:
        for result in results:
            print("===167 result:", result)
            for box, label in zip(result['boxes'], result['labels']):
                box_color = BOX_COLOR[label.item()]
                x1, x2, x3, x4 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                cv2.rectangle(frame, (x1,x2),(x3,x4), box_color, 2)
                # cv2.putText(frame, self.__class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                # cv2.putText(frame, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    cv2.polylines(frame, [np.array(normalized_road_roi_polygon)], isClosed=True, color=(32, 32, 128), thickness=2)
    return frame

def normalize_polygon(image_dim, ROAD_ROI_POLYGON):
    image_width, image_height = image_dim
    normalized_polygon = [(int(point[0] * image_width), int(point[1] * image_height)) for point in ROAD_ROI_POLYGON]
    return normalized_polygon


def stream_to_localhost(frame):
    # Encode the frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)

    # Convert the frame to bytes
    frame_bytes = buffer.tobytes()

    # Yield the frame bytes as bytes
    return (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Create a new object and execute.
if __name__ == "__main__":
    camera_ip = os.getenv("CAMERA_IP")
    if camera_ip is None:
        camera_ip = "https://d357-2a02-a313-23a-9700-a50c-d5fa-7807-1fe4.ngrok-free.app/cam-hi.jpg"
    app.run(host='0.0.0.0', port=5000, debug=True)
