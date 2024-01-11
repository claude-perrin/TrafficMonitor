from collections import Counter
from conf import *
import os
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


registry = CollectorRegistry()
car_metric = Gauge('vehicle_count', 'Number of vehicles detected', ['camera_id'], registry=registry)


def publish_inference_artifacts(outputs, camera_ip, prometheus_ip):
    if prometheus_ip is None:
        prometheus_ip = ""
    print(f"========SAVING ARTIFACT from camera: {camera_ip} to prometheus gateway: {prometheus_ip}")
    for prediction in outputs:
        artifact = parse_prediction(prediction)
        parse_metrics(artifact, camera_ip)
        publish_artifacts(prometheus_ip, camera_ip)

def parse_prediction(prediction):
    """
    Prepare artifacts to be submitted to prometheus
    """
    labels = prediction["labels"]
    labels = [IDX_TO_CLASSES[label.item()] for label in labels] 
    labels_count = Counter(labels)
    for class_name in list(CLASSES_TO_IDX.keys())[1:]:
        if class_name not in labels_count.keys():
            labels_count[class_name] = 0
    
    artifact = labels_count
    return artifact

def parse_metrics(data, camera_id):
    car_metric.labels(camera_id=camera_id).set(data.get("car", 0))

def publish_artifacts(prometheus_ip, camera_ip):
    push_to_gateway(prometheus_ip, job=f"{camera_ip}", registry=registry)

if __name__ == "__main__":
    pass
    import torch
    z = {"labels" : [torch.tensor(1), torch.tensor(1), torch.tensor(1),torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1),torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1),torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1),torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1),torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1),torch.tensor(1)]}
    # z = {"labels" : [torch.tensor(1), torch.tensor(1)]}
    # z = {"labels" : [torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1)]}

    prometheus_ip = os.getenv("PROMETHEUS_GATEWAY")
    publish_inference_artifacts([z], "127.0.0.1:8000", prometheus_ip)
