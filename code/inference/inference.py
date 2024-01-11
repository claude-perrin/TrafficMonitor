import cv2
import torchvision
import numpy as np
import torch
import urllib
from conf import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import ssd300_vgg16
import concurrent.futures
from torchvision.ops.boxes import box_iou


from process_artifacts import publish_inference_artifacts
import os
from flask import Flask, Response
from inference_helper import plot_boxes, denormalize_polygon, inference_filter_prediction, preprocess_image
from setup import InferenceSetupException, check_consistancy, get_available_models
app = Flask(__name__)

class ObjectDetection:
    """
    Class implements FasterRcnn model to make inferences.
    """

    def __init__(self, normalized_road_roi_polygon, models_name, models_weight):
        self.models_name = models_name.split()
        self.models_weight = models_weight.split()
        self.models = self.load_models()
        self.classes = list(CLASSES_TO_IDX.keys())
        self.number_of_classes = NUMBER_OF_CLASSES
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.normalized_road_roi_polygon = normalized_road_roi_polygon
    
    def load_models(self):
        models = []
        for model_name in self.models_name:
            model_path = model_name+".pth"
            if model_name == "ssd":
                models.append(self.load_ssd(model_path))
            elif model_name == "fasterrcnn1":
                models.append(self.load_fasterrcnn1(model_path))
            elif model_name == "fasterrcnn2":
                models.append(self.load_fasterrcnn2(model_path))
        return models

    def load_ssd(self, model_path):
        ssd_model = ssd300_vgg16(weights=False)
        num_anchors = ssd_model.anchor_generator.num_anchors_per_location()
        out_channels = [512,1024,512,256,256,256]
        # ssd_model.head = ssd.SSDHead(out_channels, num_anchors, num_classes+1)
        state_dict= torch.load(model_path)
        updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
        ssd_model.load_state_dict(updated_state)
        print(f"MODEL from volume: {model_path} is loaded successfuly")

        return ssd_model

    def load_fasterrcnn1(self, model_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.number_of_classes)
        state_dict= torch.load(model_path)
        updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
        model.load_state_dict(updated_state)
        print(f"MODEL from volume: {model_path} is loaded successfuly")
        return model

    def load_fasterrcnn2(self, model_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_2(weights=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.number_of_classes)
        state_dict= torch.load(model_path)
        updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
        model.load_state_dict(updated_state)
        print(f"MODEL from volume: {model_path} is loaded successfuly")
        return model

    def __call__(self, img):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        """
        grayscale = preprocess_image(img)
        grayscale = grayscale.unsqueeze(0)
        output = self.inference(grayscale)
        output = inference_filter_prediction(output, self.normalized_road_roi_polygon)
        return output


    def inference(self, input_data):
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for model in self.models:
                output = executor.submit(self.parse_data(model, input_data))
                print(f"{model} Inference output: ", output)
                futures.append(output)
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return futures
    
    def parse_data(self, model, input_data):
        model.eval()
        with torch.no_grad():
            output = model(input_data)
        return output

    def perform_ensemble_inference(self, outputs1, outputs2):
        final_prediction = []
        
        for output1, output2 in zip(outputs1, outputs2):
            if len(output1["scores"]) != 0 and len(output2["scores"]) != 0:
                output1, output2, common_preditions = self.filter_common_predictions(output1, output2)
                print("output1", output1)
                print("output3", output2)
                print("common_preditions", common_preditions)
                output1_score = output1["scores"] * 0.7
                output2_score = output2["scores"] * 0.7

                boxes = torch.cat((common_preditions["boxes"], output1["boxes"], output2["boxes"]))
                labels = torch.cat((common_preditions["labels"], output1["labels"], output2["labels"]))
                scores = torch.cat((common_preditions["scores"], output1_score , output2_score))

                predictions = {"boxes": boxes, "labels": labels, "scores": scores}
                final_prediction.append(predictions)
            else:
                final_prediction = outputs1
        return final_prediction
    
    def filter_common_predictions(self, pred_model1, pred_model2, iou_threshold=0.6):
        iou_mask = box_iou(pred_model1["boxes"], pred_model2["boxes"]) > iou_threshold

        mask_model1 = iou_mask.sum(axis=1) > iou_threshold # axis=1 is related to pred_model1
        mask_model2 = iou_mask.sum(axis=0) > iou_threshold # axis=0 is related to pred_model2
        confidences_model1 = pred_model1["scores"][mask_model1]
        confidences_model2 = pred_model2["scores"][mask_model2]
        best_confidence = torch.max(confidences_model1, confidences_model2)
        print("best_confidence: ", best_confidence)

        common_predictions = {k: v[mask_model1] for k, v in pred_model1.items()}
        common_predictions["scores"] = best_confidence
        pred_model1 = {k: v[~mask_model1] for k, v in pred_model1.items()}
        pred_model2 = {k: v[~mask_model2] for k, v in pred_model2.items()}

        return pred_model1, pred_model2, common_predictions
    
    def __class_to_label(self, x):
        return IDX_TO_CLASSES[int(x)]



@app.route('/video')
def video_feed():
    return Response(start_loop(),
                mimetype='multipart/x-mixed-replace; boundary=frame')


def start_loop():
    image_dim = (800,600)
    normalized_road_roi_polygon = denormalize_polygon(image_dim, ROAD_ROI_POLYGON)
    camera_ip = os.getenv("CAMERA_IP")
    prometheus_gateway = os.getenv("PROMETHEUS_GATEWAY")
    models_name = os.getenv("ACTIVE_MODELS")
    models_weight = os.getenv("MODELS_WEIGHT")

    detector = ObjectDetection(normalized_road_roi_polygon, models_name, models_weight)

    while True:
        img = get_image(camera_ip)
        output = detector(img)
        print("OUTPUT: ", output)
        frame = plot_boxes(output, img, normalized_road_roi_polygon)
        publish_inference_artifacts(output, camera_ip, prometheus_gateway)
        yield stream_to_localhost(frame)


def get_image(camera_ip):
    imgResp=urllib.request.urlopen(camera_ip)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    return img




def stream_to_localhost(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    return (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == "__main__":
    port = os.getenv("INFERENCE_PORT")
    try:
        if check_consistancy():
            app.run(host='0.0.0.0', port=port, debug=True)
    except InferenceSetupException as e:
        print(e)
