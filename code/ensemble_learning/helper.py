import os

# from matplotlib import pyplot as plt

import cv2
import torch
from torchvision import transforms
from dataset import ObjectDetectionDataset
import datetime
from torchvision.ops import nms
from conf import *

def custom_collate_fn(batch):
    z = datetime.datetime.now()
    targets = list()

    boxes, labels, areas, original_images, grayscale_images, idxes = parse_batch(batch)

    boxes = pad_tensors(boxes, padding_value=[0.0,0.0,1.0,1.0])
    labels = pad_tensors(labels, padding_value=CLASSES_TO_IDX["background"])
    areas = pad_tensors(areas, padding_value=1.0)

    for box, label, area, idx in zip(boxes, labels, areas, idxes):
        target = {}
        target["boxes"] = box
        target["labels"] = label
        target["area"] = area
        target["idx"] = idx
        targets.append(target)

    return grayscale_images, targets, original_images


def parse_batch(batch):
    boxes = list()
    labels = list()
    areas = list()
    original_images =  list()
    grayscale_images = list()
    idxes = list()
    for obj in batch:
        grayscale, target, original_image, idx = obj
        grayscale_images.append(grayscale)
        original_images.append(original_image)
        idxes.append(idx)

        box, label, area = target
        boxes.append(box)
        labels.append(label)
        areas.append(area)
    return boxes, labels, areas, original_images, grayscale_images, idxes


def pad_tensors(tensor_list, padding_value):
    result_tensor = []
    max_rows = max(tensor.shape[0] for tensor in tensor_list)
    padding_value = torch.tensor([padding_value])
    
    for tensor in tensor_list:
        n_rows = tensor.shape[0]
        number_to_append = max_rows - n_rows
        if number_to_append != 0:
            x_appended = torch.cat([padding_value] * number_to_append, dim=0)
            result_tensor.append(torch.cat((tensor, x_appended), dim=0))
        elif number_to_append == 0:
            result_tensor.append(tensor)
        elif number_to_append == max_rows:
            x_appended = torch.cat([padding_value] * number_to_append, dim=0)
            result_tensor.append(x_appended)
    return result_tensor


def get_train_data_loader(batch_size, training_dir, is_distributed=False, **kwargs):
    print(f"=====[INFO] Get train data loader training_dir: {training_dir}")

    transform = transforms.Compose([ transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    dataset = ObjectDetectionDataset(
        training_dir,
        transforms=transform
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    )
    print(f"=====[INFO] Got dataset {dataset}")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **kwargs
    )


def get_test_data_loader(test_batch_size, test_dir,  **kwargs):
    print(f"=====[INFO] Get test data loader test_dir: {test_dir}")
    transform = transforms.Compose([ transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    dataset = ObjectDetectionDataset(
        test_dir,
        transforms=transform
    )
    print(f"=====[INFO] Got test loader")
    print(f"============datset[0]" ,dataset[0])

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True,
        **kwargs
    )

def save_test_img(img, target, prefix):
    # img = img.permute(2,0,1).cpu().numpy()  # Convert to (height, width, channels)
    img = img.cpu().numpy()  # Convert to (height, width, channels)

    # img = img.astype('uint8')
    # img = img
    # Draw bounding boxes on the image
    for box, label in zip(target['boxes'], target['labels']):
        print(box)
        print(label)
        x, y, w, h = box.tolist()
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (w, h), BOX_COLOR[label], 2)

    # Save the image with bounding boxes
    if not os.path.exists(os.path.join(os.getcwd(), 'test_output')):
        os.makedirs(os.path.join(os.getcwd(), 'test_output'))
    img_path = f"./test_output/output_image_{prefix}.png"
    cv2.imwrite(img_path, img)
    return img_path



def create_output_bucket(s3):
    module_dir = os.environ["SM_MODULE_DIR"]
    print(f"==============BUCKET: module_dir {module_dir}")
    job_id = module_dir.split("/")[3]
    bucket_name = f'vdidyk-test-output-{job_id[-3:]}'
    print(f"==============BUCKET: {bucket_name}")
    existing_buckets = [response["Name"] for response in s3.list_buckets()["Buckets"]]
    print('Existing buckets: ', existing_buckets)
    try:
        if bucket_name not in existing_buckets:
            s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': 'eu-central-1'})
    except Exception as exc:
        print(f"Exception when creating a bucket was captured {exc}")
    return bucket_name


def save_to_bucket(img_pathes, bucket_name, s3):
    img_pathes = img_pathes[:6]
    for img_path in img_pathes:
        image_name = img_path.split("/")[-1]
        print(f"====SAVING: s3: {s3}  img_path: {img_path}  bucket_name: {bucket_name}  image_name: {image_name}")
        s3.upload_file(img_path, bucket_name, image_name)


def remove_empty_images(files_dir):
    imgs = [image for image in sorted(os.listdir(files_dir)) if image[-4:] == '.png']

    for img_name in imgs:
        image_path = os.path.join(files_dir, img_name)
        label_filename = img_name[:-4] + '.txt'
        label_file_path = os.path.join(files_dir, label_filename)

        with open(f"{label_file_path}", 'r') as f:
            txt_annotation = f.readlines()

        if not txt_annotation:
            print("remove", image_path)
            os.remove(image_path)
            os.remove(label_file_path)

def filter_prediction(predicted_dict, max_predictions, iou_threshold=0.7, confidence_threshold=0.15):
    predicted_boxes, scores = predicted_dict["boxes"], predicted_dict["scores"]
    nms_indices = nms(predicted_boxes, scores, iou_threshold)
    predicted_dict = {k: v[nms_indices] for k,v in predicted_dict.items()}

    # mask = predicted_dict["scores"] >= confidence_threshold
    # predicted_dict = {k: v[mask] for k,v in predicted_dict.items()}

    # Limit the number of predictions
    filtered_boxes, filtered_labels, filtered_scores = predicted_dict["boxes"],predicted_dict["labels"], predicted_dict["scores"]
    if len(filtered_boxes) > max_predictions:
        top_indices = torch.topk(filtered_scores, max_predictions).indices
        predicted_dict = {k: v[top_indices] for k,v in predicted_dict.items()}
    elif len(filtered_labels) < max_predictions:
        n_to_pad = max_predictions - len(filtered_labels)
        predicted_dict["labels"] = pad(filtered_labels, (0, n_to_pad), value=CLASSES_TO_IDX["background"])
        print("AFTER\n", predicted_dict["labels"])
    return predicted_dict


def clean_targets(targets):
    cleaned_targets = {}

    # Filter out invalid boxes
    valid_boxes_mask = (targets['boxes'].sum(axis=1) > 2)    
    valid_labels_mask = (targets['labels'] != CLASSES_TO_IDX["background"])
    valid_area_mask = (targets['area'] >= 1)

    # Combine all conditions using "&"
    final_mask = valid_boxes_mask & valid_labels_mask & valid_area_mask
    print(final_mask)
    # Apply the final mask
    for key, target_tensor in targets.items():
        if key == "idx":
            cleaned_targets[key] = target_tensor
            continue
        cleaned_targets[key] = target_tensor[final_mask]
    return cleaned_targets

def save_model(model, model_dir, model_prefix):
    print(f"Saving model: {model} \n\n Saving to model_dir: {model_dir}")
    path = os.path.join(model_dir, f"{model_prefix}_model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    return path