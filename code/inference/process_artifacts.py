from collections import Counter
from conf import IDX_TO_CLASSES
import os
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


registry = CollectorRegistry()
car_metric = Gauge('car_count', 'Number of car detected', ['camera_id'], registry=registry)
bus_metric = Gauge('bus_count', 'Number of buses detected', ['camera_id'], registry=registry)
cyclist_metric = Gauge('cyclist_count', 'Number of cyclists detected', ['camera_id'], registry=registry)
pedestrian_metric = Gauge('pedestrian_count', 'Number of pedestrians detected', ['camera_id'], registry=registry)



def parse_prediction(prediction):
    """
    Prepare artifacts to be submitted to prometheus
    """
    labels = prediction["labels"]
    labels = [IDX_TO_CLASSES[label.item()] for label in labels] 
    artifact = Counter(labels)
    return artifact

def parse_metrics(data, camera_id):
    car_metric.labels(camera_id=camera_id).set(data.get("vehicle", 0))
    bus_metric.labels(camera_id=camera_id).set(data.get("bus", 0))
    cyclist_metric.labels(camera_id=camera_id).set(data.get("cyclist", 0))
    pedestrian_metric.labels(camera_id=camera_id).set(data.get("pedestrian", 0))

def publish_artifacts(prometheus_ip, camera_ip):
    push_to_gateway(prometheus_ip, job=f"{camera_ip}", registry=registry)

def process_inference_output(outputs, camera_ip, prometheus_gateway):
    if prometheus_gateway is None:
        prometheus_gateway = ""
    print(f"========SAVING ARTIFACT from camera: {camera_ip} to prometheus gateway: {prometheus_gateway}")
    for prediction in outputs:
        artifact = parse_prediction(prediction)
        parse_metrics(artifact, camera_ip)
        publish_artifacts(prometheus_gateway, camera_ip)

if __name__ == "__main__":
    pass
    import torch
    z = {"labels" : [torch.tensor(4), torch.tensor(4), torch.tensor(4),torch.tensor(4), torch.tensor(4), torch.tensor(4), torch.tensor(4),torch.tensor(4)]}
    # z = {"labels" : [torch.tensor(4), torch.tensor(4), torch.tensor(4), torch.tensor(4),torch.tensor(4)]}
    prometheus_ip = os.getenv("PROMETHEUS_GATEWAY")
    process_inference_output([z], "127.0.0.1:8000", prometheus_ip)
