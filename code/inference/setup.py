import os
import requests

class InferenceSetupException(Exception):
    """
    Raised when during setup, machine learning models were not found 
    inside /app/inference/model dir
    """
    pass


def check_models_volume(model_dir_name):
    if os.path.isdir(model_dir_name):
        return True
    raise InferenceSetupException(f"Directory {model_dir_name} does not exist.. \nExiting the inference")

def check_models_presence(model_dir_name, model_name):
    # for model_name in models:
    if not os.path.isfile(f"{model_dir_name}/{model_name}.pth"):
        raise InferenceSetupException(f"Model: {model_name} does not exist.. inside {model_dir_name}\nExiting the inference")
    return True

# def check_weight_consistency(models_name, models_weight):
#     if len(models_name) != len(models_weight):
#         raise InferenceSetupException(f"Inconsistency in specified models : {models_name} and weights {models_weight}..\nExiting the inference")
#     return True

def get_available_models(model_dir_name):
    available_models = os.path.listdir(model_dir_name)
    if len(available_models) == 0:
        raise InferenceSetupException(f"Directory {model_dir_name} is empty.. \nExiting the inference")
    return available_models


def check_endpoint(endpoint):
    try:
        response = requests.get(endpoint)
        return response.raise_for_status()
    except requests.RequestException as e:
        raise InferenceSetupException(f"Endpoint {endpoint} cannot be reached.. \ndue to {e}\nExiting the inference")


def check_consistancy():
    model_dir_name = os.getenv("MODELS_DOCKER_PATH")
    models_name = os.getenv("ACTIVE_MODELS")
    models_weight = os.getenv("MODELS_WEIGHT")
    camera_endpoint = os.getenv("CAMERA_IP")
    prometheus_gateway_endpoint = os.getenv('PROMETHEUS_GATEWAY')

    try:
        check_models_volume(model_dir_name)
        check_models_presence(model_dir_name, models_name)
        # check_weight_consistency(models_name, models_weight)
        # check_endpoint(camera_endpoint)
        # check_endpoint(prometheus_gateway_endpoint)
    except InferenceSetupException as e:
        raise InferenceSetupException(e)
    return True
