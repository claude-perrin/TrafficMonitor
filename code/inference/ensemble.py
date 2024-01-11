from multiprocessing import Pool, TimeoutError
from functools import partial
import time
import torch
from conf import ROAD_ROI_POLYGON
import sys
import os

from torchvision.ops.boxes import box_iou


class Ensemble:
    def __init__(self, estimators):
        self._estimators = estimators
    
    def eval(self):
        for i in self._estimators:
            i.eval()
   
    @property
    def name(self):
        return " ".join([i.name for i in self._estimators])
    
    def __call__(self, data):
        # image_dim = data[0].shape
        # image_dim = (image_dim[2],image_dim[1])
        # roi = denormalize_polygon(image_dim, ROAD_ROI_POLYGON)

        start = time.time()
        with Pool(processes=5) as pool:
            partial_func = partial(self.predict, data=data)
            results = pool.map(partial_func, self._estimators)
        end = time.time() - start
        print("TIME TAKEN: ", end)
        
        # predictions = {key: inference_filter_prediction(prediction) for key, prediction in results}
        predictions = {key: prediction for key, prediction in results}

        print("predictions: ", predictions)
        return predictions

    def filter_predictions(self, predictions, iou_threshold=0.25, roi_polygon=None, **kwargs):
        filtered_predictions = {}
        for estimator in self._estimators:
            estimators_prediction = predictions[estimator.name]
            estimator_confidence_threshold = estimator.confidence_threshold
            estimators_prediction = estimator.filter_predictions(estimators_prediction, confidence_threshold=estimator_confidence_threshold, iou_threshold=iou_threshold, roi_polygon=roi_polygon)
            filtered_predictions[estimator.name] = estimators_prediction
        ensembled_prediction = self.run_ensemble(filtered_predictions)
        print("ensembled_prediction", ensembled_prediction)
        # [box: votes]
        return ensembled_prediction

    @staticmethod
    def predict(estimator, data):
        with torch.no_grad():
            return (estimator.name, estimator(data))

    def run_ensemble(self, predictions):
        models = tuple(predictions.keys())
        print("models", models)
        zipped_predictions = list(zip(*predictions.values()))
        ensembled_predictions = []
        for predictions in zipped_predictions:
            common_boxes = self.ensemble_common_boxes(predictions=predictions, models=models)
            ensembled_prediction = self.bagging_voting(common_boxes)
            ensembled_predictions.append(ensembled_prediction)
        return ensembled_predictions

    def ensemble_common_boxes(self, predictions, models, iou_threshold=0.6):
        common_boxes_result = {}
        for i in range(len(predictions[0]["boxes"])):
            key_box = tuple(predictions[0]["boxes"][i].tolist())
            model = [models[0]]
            scores = predictions[0]["scores"][i].unsqueeze(0)
            labels = predictions[0]["labels"][i].unsqueeze(0)
            common_boxes_result[key_box] = {"models": model, "scores": scores, "labels": labels}
        comparison_tensor = predictions[0]["boxes"]
        if len(comparison_tensor) == 0:
            comparison_tensor = torch.tensor([[1,1,1,1]])
        print("predictions: ", predictions)
        for model_id, prediction in enumerate(predictions[1:], 1):
            current_model = models[model_id]
            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]

            iou_mask = box_iou(comparison_tensor, boxes)
            if len(iou_mask) > 1:
                mask1 = iou_mask.max(axis=1).values > iou_threshold
                mask2 = iou_mask.max(axis=0).values > iou_threshold
                print("comparison_tensor: ", comparison_tensor)
                print("boxes: ", boxes)
                print("mask1: ", mask1)
                print("mask2: ", mask2)

                matching_boxes = comparison_tensor[mask1]
                new_boxes = boxes[~mask2]
                print("matching_boxes: ", matching_boxes)
                print("new_boxes: ", new_boxes)
                common_boxes_result = self.modify_ensemble_common_boxes(common_boxes_result, current_model, matching_boxes, scores, labels, mask2)
                if len(new_boxes):
                    comparison_tensor = torch.cat((comparison_tensor, new_boxes))
                    common_boxes_result = self.modify_ensemble_common_boxes(common_boxes_result, current_model, new_boxes, scores, labels, ~mask2)

        print("common_boxes_result: ",common_boxes_result)
        return common_boxes_result # {[box] : {models: [models], labels: [labels], "scores": [scores]}}
    
    @staticmethod
    def modify_ensemble_common_boxes(common_boxes_result, current_model, boxes, scores, labels, mask):
        new_scores = scores[mask]
        new_labels = labels[mask]
        print("new_score: ", new_scores)
        print("new_label: ", new_labels)

        for box, score, label in zip(boxes, new_scores, new_labels):
            box = tuple(box.tolist())
            score = torch.tensor(score).unsqueeze(0)
            label = torch.tensor(label).unsqueeze(0)
            if box not in common_boxes_result:
                common_boxes_result[box] = {"models": [current_model], "scores": score, "labels": label}
            else:
                common_boxes_result[box]["models"].append(current_model)
                common_boxes_result[box]["scores"] = torch.cat((common_boxes_result[box]["scores"], score), dim=0)
                common_boxes_result[box]["labels"] = torch.cat((common_boxes_result[box]["labels"], label), dim=0)
        return common_boxes_result


    def bagging_voting(self, common_boxes_result):
        boxes = []
        scores = []
        labels = []
        for common_box, predictions in common_boxes_result.items():
            prediction_number = len(predictions["models"])
            ensembled_prediction_rate = prediction_number/len(self._estimators)
            if ensembled_prediction_rate >= 0.5:
                boxes.append(common_box)
                score = torch.max(predictions["scores"]).item()
                label = predictions["labels"][0].item()
                scores.append(score)
                labels.append(label)
        ensembled_prediction = {"boxes": torch.tensor(boxes), "scores": torch.tensor(scores), "labels": torch.tensor(labels)}
        return ensembled_prediction
