from collections import Counter
from conf import *
import json
from datetime import datetime
import os
artifacts_path = "../artifacts"

def parse_artifacts(output):
    """
    Prepare artifacts to be submitted to prometheus
    """
    prediction = output
    labels = prediction["labels"]
    labels = [IDX_TO_CLASSES[label.item()] for label in labels] 
    labels_count = Counter(labels)
    for class_name in list(CLASSES_TO_IDX.keys())[1:]:
        if class_name not in labels_count.keys():
            labels_count[class_name] = 0
    
    artifact = labels_count
    return artifact
    

def publish_artifacts_locally(artifacts):
    dt_string = datetime.now().strftime("%Y%m%dT%H%M%S")
    dump = json.dumps(artifacts, indent=4)
    with open(f"{artifacts_path}/{dt_string}.json", "w+") as f:
       f.write(dump) 

def publish_artifacts_remotely(artifacts):
    pass

def process_inference_output(outputs, camera_ip):
    camera_ip = os.getenv("CAMERA_IP")
    if camera_ip is None:
        camera_ip = "https://d357-2a02-a313-23a-9700-a50c-d5fa-7807-1fe4.ngrok-free.app/cam-hi.jpg"
    print(f"========SAVING ARTIFACTS======== from camera: {camera_ip}")
    for output in outputs:
        artifacts = parse_artifacts(output)
    # publish_artifacts_locally(artifacts)

