import cv2
import torchvision
import numpy as np
import torch
import urllib
from conf import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from process_artifacts import process_inference_output
import os
from flask import Flask, Response
from inference_helper import normalize_polygon, inference_filter_prediction, preprocess_image

app = Flask(__name__)

class ObjectDetection:
    """
    Class implements FasterRcnn model to make inferences.
    """

    def __init__(self, normalized_road_roi_polygon, model_path="models/FasterRcnn_V1_epoch-4_model.pth"):
        self.model_path = model_path
        self.model = self.load_model()
        self.classes = list(CLASSES_TO_IDX.keys())
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.normalized_road_roi_polygon = normalized_road_roi_polygon
    
    def load_model(self, num_classes=NUMBER_OF_CLASSES):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        state_dict= torch.load(self.model_path)
        updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
        model.load_state_dict(updated_state)
        print(f"MODEL from volume: {self.model_path} is loaded successfuly")
        return model

    def __call__(self, img):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        """
        grayscale = preprocess_image(img)
        grayscale = grayscale.unsqueeze(0)
        output = self.__inference(self.model, grayscale)
        output = inference_filter_prediction(output, self.normalized_road_roi_polygon)
        return output

    def __inference(self, model, input_data):
        model.eval()
        with torch.no_grad():
            output = model(input_data)
        print("Inference output: ", output)
        return output

    def __class_to_label(self, x):
        return IDX_TO_CLASSES[int(x)]



@app.route('/video')
def video_feed():
    return Response(start_loop(),
                mimetype='multipart/x-mixed-replace; boundary=frame')

def start_loop():
    image_dim = (800,600)
    normalized_road_roi_polygon = normalize_polygon(image_dim, ROAD_ROI_POLYGON)
    a = ObjectDetection(normalized_road_roi_polygon)
    camera_ip = os.getenv("CAMERA_IP")
    prometheus_gateway = os.getenv("PROMETHEUS_GATEWAY")

    while True:
        img = get_image(camera_ip)
        output = a(img)
        print("OUTPUT: ", output)
        frame = plot_boxes(normalized_road_roi_polygon, output, img)
        process_inference_output(output, camera_ip, prometheus_gateway)
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
    label_bg_white = (255, 255, 255)
    if len(results) != 0:
        for result in results:
            print(f"===160 {result}")
            for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
                label = label.item()
                score = score.item()
                box_color = BOX_COLOR[label]
                x1, x2, x3, x4 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                cv2.rectangle(frame, (x1,x2),(x3,x4), box_color, 2)
                cv2.rectangle(frame, (x1, x2-25), (x1+150, x2), label_bg_white, -1)
                label_text = f'{class_to_label(label)}: {score:.2f}'
                cv2.putText(frame, label_text, (x1, x2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
    cv2.polylines(frame, [np.array(normalized_road_roi_polygon)], isClosed=True, color=(32, 32, 128), thickness=2)
    return frame

def class_to_label(label):
    return IDX_TO_CLASSES[label]


def stream_to_localhost(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    return (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == "__main__":
    camera_ip = os.getenv("CAMERA_IP")
    port = os.getenv("INFERENCE_PORT")
    if camera_ip is None:
        print(f"CAMERA_IP is not SPECIFIED: \ncCAMERA_IP: '{camera_ip}'\nPROMETHEUS_GATEWAY: {os.getenv('PROMETHEUS_GATEWAY')}")
    app.run(host='0.0.0.0', port=port, debug=True)
