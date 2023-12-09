from inference_helper import normalize_polygon, inference_filter_prediction
from conf import ROAD_ROI_POLYGON
from helper import get_train_data_loader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from conf import *
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
        output = self.__inference(self.model, img)
        output = inference_filter_prediction(output, self.normalized_road_roi_polygon)
        return output

    def __inference(self, model, input_data):
        model.eval()
        with torch.no_grad():
            output = model(input_data)
        print("Inference output: ", output)
        return output


if __name__ == "__main__":
    device = torch.device("cpu")
    train_loader = get_train_data_loader(4, "./real_frames")
    image_dim = (960,1280)
    normalized_road_roi_polygon = normalize_polygon(image_dim, ROAD_ROI_POLYGON)
    model = ObjectDetection(normalized_road_roi_polygon, model_path="models/FasterRcnn_V1_epoch-4_model.pth")
    
    for batch_idx, (data, targets, original_image) in enumerate(train_loader, 1):
        output = model(data)
        print(output)
