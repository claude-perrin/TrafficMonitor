from ensemble import Ensemble
from estimators import FasterRcnn, SSD, FasterRcnn_MobileNet, FasterRcnn2, RetinaNet
import torch
import os
import sys
from conf import ROAD_ROI_POLYGON
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))

from inference_helper import inference_filter_prediction, denormalize_polygon, plot_boxes
from helper import get_test_data_loader
import cv2

# ensemble #1 is faster_rcnn, ssd, retina, faster_rcnn2, rcnn_mobile
# ensemble #2 is faster_rcnn, ssd, retina
if __name__ == "__main__":
    current_path = os.path.abspath(os.path.dirname(__file__))
    faster_rcnn_path = f"{current_path}/../models/Good_FasterRcnn_V1_model.pth"
    faster_rcnn_path2 = f"{current_path}/../models/Good_FasterRcnn_V2_model.pth"

    ssd_path = f"{current_path}/../models/Good_SSD_model.pth"
    ssd = SSD(ssd_path)
    ssd_default = SSD(None)

    faster_rcnn = FasterRcnn(faster_rcnn_path)
    rcnn_mobile = FasterRcnn_MobileNet(None)
    retina = RetinaNet(None)
    faster_rcnn_2 = FasterRcnn2(faster_rcnn_path2)
    dataloader = get_test_data_loader(2, f"{current_path}/../test/test_data/")
    image_dim = (960,1280)
    denormalized_road_roi_polygon = denormalize_polygon(image_dim, ROAD_ROI_POLYGON)

    # estimators = [ssd, faster_rcnn, faster_rcnn_2]
    estimators = [faster_rcnn, ssd, retina]
    ensemble = Ensemble(estimators)
    # ensemble = retina 
    ensemble.eval()
    ensembled_result = []
    for batch_idx, (data, targets,original_images) in enumerate(dataloader, 1):
        predictions = ensemble(data)
        filtered_result = ensemble.filter_predictions(predictions, roi_polygon=denormalized_road_roi_polygon, confidence_threshold=0.3)
        ensembled_result.append(filtered_result)
        frame = plot_boxes(filtered_result[0], original_images[0].numpy(), denormalized_road_roi_polygon)
        print("Showing image")
        cv2.imshow("die", frame)
        cv2.imwrite("./image.jpg", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
            