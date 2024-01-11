from abc import ABC
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn,fasterrcnn_resnet50_fpn_v2, ssd300_vgg16, retinanet_resnet50_fpn_v2
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights, SSD300_VGG16_Weights, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from inference_helper import inference_filter_prediction
import yolov5
import numpy
import cv2

NUMBER_OF_CLASSES = 5

class LearningModel(ABC):
    _confidence_threshold = 0.5

    def __init__(self, _weight_path):
        self._weight_path = _weight_path
        self._model = self.load_model()

    def __call__(self, data):
        pass

    def load_model(self, num_classes):
        pass   

    def eval(self):
        self._model.eval()
    
    def filter_predictions(self, predictions, confidence_threshold, iou_threshold=0.25, roi_polygon=None):
        filtered_predictions = inference_filter_prediction(predictions, confidence_threshold=confidence_threshold, iou_threshold=iou_threshold, roi_polygon=roi_polygon)
        return filtered_predictions
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def confidence_threshold(self):
        return self._confidence_threshold
    

class FasterRcnn(LearningModel):
    _confidence_threshold = 0.7

    def __call__(self, data):
        with torch.no_grad():
            predictions = self._model(data)
        for prediction in predictions:
            prediction["labels"] = torch.where(prediction["labels"] == 4, 1, prediction["labels"])
            prediction["labels"] = torch.where(prediction["labels"] == 3, 2, prediction["labels"])
        return predictions
    
    def load_model(self, num_classes=NUMBER_OF_CLASSES):
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        state_dict= torch.load(self._weight_path)
        updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
        model.load_state_dict(updated_state)
        print(f"MODEL from volume: {self._weight_path} is loaded successfuly")
        return model
        
class FasterRcnn2(LearningModel):
    _confidence_threshold = 0.7

    def __call__(self, data):
        with torch.no_grad():
            predictions = self._model(data)
        for prediction in predictions:
            prediction["labels"] = torch.where(prediction["labels"] == 4, 1, prediction["labels"])
            prediction["labels"] = torch.where(prediction["labels"] == 3, 2, prediction["labels"])
        return predictions
    
    def load_model(self, num_classes=NUMBER_OF_CLASSES):
        model = fasterrcnn_resnet50_fpn_v2(weights=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        state_dict= torch.load(self._weight_path)
        updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
        model.load_state_dict(updated_state)
        print(f"MODEL from volume: {self._weight_path} is loaded successfuly")
        return model

class FasterRcnn_MobileNet(LearningModel):
    def __call__(self, data):
        with torch.no_grad():
            predictions = self._model(data)
        for prediction in predictions:
            prediction["labels"] = torch.where(prediction["labels"] == 4, 1, prediction["labels"])
            prediction["labels"] = torch.where(prediction["labels"] == 3, 1, prediction["labels"])
        return predictions
    
    def load_model(self, num_classes=NUMBER_OF_CLASSES):
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        print(f"MODEL from volume: {self._weight_path} is loaded successfuly")
        return model



class SSD(LearningModel):
    _confidence_threshold = 0.3

    def __call__(self, data):
        with torch.no_grad():
            predictions = self._model(data)
        for prediction in predictions:
            prediction["labels"] = torch.where(prediction["labels"] == 4, 1, prediction["labels"])
            prediction["labels"] = torch.where(prediction["labels"] == 3, 1, prediction["labels"])
        return predictions
    
    def load_model(self, num_classes=NUMBER_OF_CLASSES):
        if self._weight_path is not None:
            ssd_model = ssd300_vgg16(weights=None)
            num_anchors = ssd_model.anchor_generator.num_anchors_per_location()
            out_channels = [512,1024,512,256,256,256]
            # ssd_model.head = ssd.SSDHead(out_channels, num_anchors, num_classes+1)
            state_dict= torch.load(self._weight_path)
            updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
            ssd_model.load_state_dict(updated_state)
            print(f"MODEL from volume: {self._weight_path} is loaded successfuly")
        else:
            ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        return ssd_model

class RetinaNet(LearningModel):
    def __call__(self, data):
        with torch.no_grad():
            predictions = self._model(data)
        for prediction in predictions:
            prediction["labels"] = torch.where(prediction["labels"] == 4, 1, prediction["labels"])
            prediction["labels"] = torch.where(prediction["labels"] == 3, 1, prediction["labels"])
        return predictions
    
    def load_model(self, num_classes=NUMBER_OF_CLASSES):
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        print(f"MODEL from volume: {self._weight_path} is loaded successfuly")
        return model
   