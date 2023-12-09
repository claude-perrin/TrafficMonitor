from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 
from torchvision.models.detection import fasterrcnn_resnet50_fpn 
import torch
from conf import NUMBER_OF_CLASSES
from helper import get_train_data_loader, save_test_img
import datetime
from torchvision.models.detection import ssd300_vgg16

model_path1 = "models/FasterRcnn_V1_epoch-4_model.pth"
model_path2 = "models/FasterRcnn_V2_epoch-4_model.pth"
model_path3 = "models/SSD_epoch-8_model.pth"


def load_model1(num_classes=NUMBER_OF_CLASSES):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state_dict= torch.load(model_path1)
    updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
    model.load_state_dict(updated_state)
    print(f"MODEL from volume: {model_path1} is loaded successfuly")
    return model

def load_model2(num_classes=NUMBER_OF_CLASSES):
    model = fasterrcnn_resnet50_fpn_v2(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state_dict= torch.load(model_path2)
    updated_state = {k.replace("module.", ""): v for k,v in state_dict.items()}
    model.load_state_dict(updated_state)
    print(f"MODEL from volume: {model_path2} is loaded successfuly")
    return model

def load_model3(num_classes=NUMBER_OF_CLASSES):
    ssd_model = ssd300_vgg16(pretrained=True)
    classification_head = ssd_model.head.classification_head
    freeze_layers = [
        ssd_model.backbone.features,    # Freeze the VGG16 backbone
        ssd_model.backbone.extra,       # Optionally, freeze extra layers
        ssd_model.anchor_generator,     # Freeze the anchor generator
    ]
    for id, layer in enumerate(classification_head.module_list):
        new_layer = torch.nn.Conv2d(layer.in_channels, num_classes+1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        classification_head.module_list[id] = new_layer
    for layer in freeze_layers:
        for param in layer.parameters():
            param.requires_grad = False
    return ssd_model

def ensemble_inference(model1, model2, model3, dataloader, device=torch.device("cpu")):
    model1.eval()
    model2.eval()

    all_predictions = []

    with torch.no_grad():
        for idx, [inputs, _, original_images] in enumerate(dataloader):
            timer_model1 = datetime.datetime.now()
            outputs1 = model1(inputs)
            timer_model1 = datetime.datetime.now() - timer_model1 

            # print(f"outputs1: {outputs1}")
            timer_model2 = datetime.datetime.now()
            outputs2 = model2(inputs)
            timer_model2 = datetime.datetime.now() - timer_model2

            timer_model3 = datetime.datetime.now()
            outputs3 = model3(inputs)
            timer_model3 = datetime.datetime.now() - timer_model3


            result = {"model1": {"output" : outputs1, "timer": str(timer_model1)},
                      "model2": {"output" : outputs2, "timer": str(timer_model2)},
                      "model3": {"output" : outputs3, "timer": str(timer_model3)},
                      "original_images": original_images}
            all_predictions.append(result)
            print(f"Processed {idx} from {len(dataloader)}")
            break
        for idx, result in enumerate(all_predictions):
            outputs1, timer1 = result["model1"].values()
            outputs2, timer2 = result["model2"].values()
            outputs3, timer3 = result["model3"].values()
            original_images = result["original_images"]
            print(f"Time\nModel1 (Rcnn_V1): {timer1}\nModel2 (Rcnn_V2): {timer2}\nModel3 (SSD): {timer3}")
            name_prefix = f"_output_{idx}"
            for output1, output2, output3, original_image in zip(outputs1, outputs2,outputs3,original_images):
                save_test_img(torch.clone(original_image), output1, f"model1_{name_prefix}")
                save_test_img(torch.clone(original_image), output2, f"model2_{name_prefix}")
                save_test_img(original_image, output3, f"model3_{name_prefix}")


def run():
    batch_size = 2
    training_dir = "./cropped_train"
    model1 = load_model1()
    model2 = load_model2()
    model3 = load_model3()
    dataloader = get_train_data_loader(batch_size, training_dir)

    ensemble_inference(model1, model2, model3, dataloader)



if __name__ == "__main__":
    run()
