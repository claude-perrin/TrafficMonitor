import sys
import os
import torch
from torchvision.ops import box_iou
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../inference')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))

from ensemble import Ensemble
from estimators import FasterRcnn, SSD, Yolov5, FasterRcnn2, FasterRcnn_MobileNet
from inference_helper import inference_filter_prediction
from helper import get_test_data_loader
from logger import write_model_peformance, write_ensemble_model_peformance
import random


def inference(model, dataloader):
    model_name = model.name
    # confidence_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.5, 0.6, 0.7]
    # cars_confidence_accuracy = {confidence: [] for confidence in confidence_thresholds}
    # peds_confidence_accuracy = {confidence: [] for confidence in confidence_thresholds}
    cars_result = []
    peds_result = []
    for batch_idx, (data, targets, _) in enumerate(dataloader, 1):
        with torch.no_grad():
            output = model(data)
        # for conf_threshold in confidence_thresholds:
            # predictions = model.filter_predictions(output, confidence_threshold=conf_threshold)
        predictions = model.filter_predictions(output)
        for prediction, target in zip(predictions, targets):
            print(f"Batch #{batch_idx} out of {len(dataloader)}...")
            print(f"prediction: {prediction}")
            print(f"target: {target}")

            prediction_mask_cars = (prediction["labels"] == 1.)
            target_mask_cars = (target["labels"] == 1.)
            prediction_mask_ped = (prediction["labels"] == 2.)
            target_mask_ped = (target["labels"] == 2.)
            if len(prediction["boxes"]) > 1:
                cars_match_mask = box_match(prediction["boxes"][prediction_mask_cars], target["boxes"][target_mask_cars])
                ped_match_mask = box_match(prediction["boxes"][prediction_mask_ped], target["boxes"][target_mask_ped])
                print(f"cars_match_mask: {cars_match_mask}")
                # print(f"ped_match_mask: {ped_match_mask}")
                cars_iou_matrix = generate_confusion_matrix(cars_match_mask)
                ped_iou_matrix = generate_confusion_matrix(ped_match_mask)
                if len(cars_iou_matrix):
                    cars_accuracy = cars_iou_matrix[0]/sum(cars_iou_matrix) if sum(cars_iou_matrix) else 1.0
                    print(f"**==** Cars confusion Matrix: {cars_iou_matrix}  | accuracy: {cars_accuracy} | confidence_threshold: {None}" )
                    cars_result.append(cars_accuracy)
                if len(ped_iou_matrix):
                    peds_accuracy = ped_iou_matrix[0]/sum(ped_iou_matrix) if sum(ped_iou_matrix) else 1.0
                    print(f"**==** Cars confusion Matrix: {ped_iou_matrix}  | accuracy: {peds_accuracy} | confidence_threshold: {None}")
                    peds_result.append(peds_accuracy)

        # if batch_idx in range(10, 100, 10):
        write_ensemble_model_peformance(f"./log", model_name, batch_idx, cars_result, peds_result )



def generate_confusion_matrix(iou_matrix):
    answers_ground_truth = iou_matrix.sum(axis=1)

    print("iou_matrix \n", iou_matrix)
    correct_boxes_prediction = answers_ground_truth.sum().item()
    not_predicted_boxes = len(answers_ground_truth) - answers_ground_truth.sum().item()

    answers_prediction = iou_matrix.sum(axis=0)
    excessive_boxes = len(answers_prediction) - answers_prediction.sum().item()

    return (correct_boxes_prediction, excessive_boxes, not_predicted_boxes, 0)

def box_match(prediction, target, iou_match_threshold=0.55):
    iou = box_iou(prediction, target)
    print("iou:", iou)
    iou_matrix = torch.zeros(len(prediction), len(target))
    iou_matrix[iou > iou_match_threshold] = 1
    
    # mask = iou_result.sum(axis=1) > iou_match_threshold
    return iou_matrix




if __name__ == "__main__":
    current_path = os.path.abspath(os.path.dirname(__file__))
    faster_rcnn_path = f"{current_path}/../models/Good_FasterRcnn_V1_model.pth"
    ssd_path = f"{current_path}/../models/Good_SSD_model.pth"
    yolov5_path = f"{current_path}/../models/yolov5.pt"
    faster_rcnn2_path = f"{current_path}/../models/Good_FasterRcnn_V2_model.pth"
    images_path = f"{current_path}/../../test_set/"
    test_batch_size = 8

    # rcnn = FasterRcnn(faster_rcnn_path)
    ssd = SSD(None)
    # rcnn2 = FasterRcnn2(faster_rcnn2_path)
    # yolov5 = Yolov5(yolov5_path)
    rcc_mobile = FasterRcnn_MobileNet(None)
    test_dataloader = get_test_data_loader(test_batch_size, images_path)

    current_path = os.path.abspath(os.path.dirname(__file__))
    faster_rcnn_path = f"{current_path}/../models/Good_FasterRcnn_V1_model.pth"
    faster_rcnn_path2 = f"{current_path}/../models/Good_FasterRcnn_V2_model.pth"

    ssd_path = f"{current_path}/../models/Good_SSD_model.pth"
    ssd = SSD(ssd_path)
    ssd_default = SSD(None)

    faster_rcnn = FasterRcnn(faster_rcnn_path)
    rcnn_mobile = FasterRcnn_MobileNet(None)
    faster_rcnn_2 = FasterRcnn2(faster_rcnn_path2)
    # test_dataloader = get_test_data_loader(5, f"{current_path}/../test/test_data/")

    estimators = [ssd, faster_rcnn, rcnn_mobile]
    ensemble = Ensemble(estimators)
    ensemble.eval()


    inference(ensemble, test_dataloader)