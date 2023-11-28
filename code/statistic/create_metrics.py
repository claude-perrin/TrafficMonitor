from prometheus_client import start_http_server, Gauge, CollectorRegistry
import time
import os
import json

registry = CollectorRegistry()
car_metric = Gauge('car_count', 'Number of car detected', ['camera_id'], registry=registry)
bus_metric = Gauge('bus_count', 'Number of buses detected', ['camera_id'], registry=registry)
cyclist_metric = Gauge('cyclist_count', 'Number of cyclists detected', ['camera_id'], registry=registry)
pedestrian_metric = Gauge('pedestrian_count', 'Number of pedestrians detected', ['camera_id'], registry=registry)


def parse_metrics(metric_path):
    print("Parsing metric at path: ", metric_path)
    with open(metric_path, "r") as f:
        metric = json.load(f)
        print("Metric json: ", metric)
    data = metric[0]
    camera_id = metric[1]["camera_id"]

    car_metric.labels(camera_id=camera_id).set(data.get("vehicle", 0))
    bus_metric.labels(camera_id=camera_id).set(data.get("bus", 0))
    cyclist_metric.labels(camera_id=camera_id).set(data.get("cyclist", 0))
    pedestrian_metric.labels(camera_id=camera_id).set(data.get("pedestrian", 0))


def local_parse():
    current_workspace = os.getcwd()
    artifact_path = os.path.join(current_workspace, "../artifacts")

    existing_artifacts = os.listdir(artifact_path)
    print("Main, existing artifacts: ", existing_artifacts)
    while True:
        pooling_artifacts = os.listdir(artifact_path)
        added_artifacts = [f for f in pooling_artifacts if f not in existing_artifacts and f != ".DS_Store"]
        print("ADDED ARTIFACTS: ", added_artifacts)
        if added_artifacts:
            for artifact in added_artifacts:
                artifact = os.path.join(artifact_path, artifact) 
                parse_metrics(artifact)
            existing_artifacts += added_artifacts
                

        time.sleep(2)


def s3_parse():
    pass

# Expose metrics on /metrics endpoint
if __name__ == "__main__":
    start_http_server(8000, registry=registry)
    local_parse()




