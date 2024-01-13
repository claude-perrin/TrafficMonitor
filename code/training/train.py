import argparse
import os
import boto3


#import sagemaker_containers
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from helper import save_model, clean_targets, save_test_img, save_to_bucket, create_output_bucket, get_train_data_loader, get_test_data_loader, filter_prediction
import json
from torch.optim.lr_scheduler import StepLR

from conf import * 
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import ssd, ssd300_vgg16, SSD300_VGG16_Weights
def get_fasterrcnn_detection_model(num_classes=NUMBER_OF_CLASSES):
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_ssd_detection_model(num_classes=NUMBER_OF_CLASSES):
    ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    # freeze_layers = [
    #     ssd_model.backbone.features,    # Freeze the VGG16 backbone
    #     ssd_model.backbone.extra,       # Optionally, freeze extra layers
    #     ssd_model.anchor_generator,     # Freeze the anchor generator
    # ]
    num_anchors = ssd_model.anchor_generator.num_anchors_per_location()
    out_channels = [512,1024,512,256,256,256]
    ssd_model.head = ssd.SSDHead(out_channels, num_anchors, num_classes)
    for layer in ssd_model.backbone.features[:-13]:
        for param in layer.parameters():
            param.requires_grad = False
        
    return ssd_model


class Main:
    def __init__(self, args):
        self.args =  args
        self.s3_session = boto3.client('s3')
        self.image_size = (704,704)
        self.rank = 0
        self.use_cuda = args.num_gpus > 0
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # self.device = torch.device("mps")
        self.is_distributed = len(self.args.hosts) > 1 and self.args.backend is not None

    def get_model(self):
        if self.args.model == "fasterrcnn":
            model = get_fasterrcnn_detection_model(num_classes=NUMBER_OF_CLASSES).to(self.device)
        elif self.args.model == "ssd":
            model = get_ssd_detection_model(num_classes=NUMBER_OF_CLASSES).to(self.device)
        model.to(self.device)
        return model

    def setup_distributed_system(self, args):
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        print(f"==========SETUP DIST host_rank: {host_rank}")

        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        print(f"==========SETUP DIST args.backend {args.backend} ; dist.get_world_size: {dist.get_world_size()}")

    def run(self):
        if self.is_distributed:
            self.setup_distributed_system(self.args)
            self.rank = dist.get_rank()
        torch.manual_seed(self.args.seed)
        if self.use_cuda:
            torch.cuda.manual_seed(self.args.seed)
        if self.rank == 0:
            pass #TODO
            self.output_bucket_name = create_output_bucket(self.s3_session)
        kwargs = {"num_workers": 1, "pin_memory": True} if self.use_cuda else {}

        model = self.get_model()

        print(f"=====[INFO] Traing loop variables:\n is_distributed: {self.is_distributed}; device: {self.device}\n")
        train_loader = get_train_data_loader(self.args.batch_size, self.args.train_dir, is_distributed=self.is_distributed, transform=True, **kwargs)
        # test_loader = get_test_data_loader(self.args.test_batch_size, self.args.test_dir, **kwargs)
        print(f"len(train_loader.sampler), len(train_loader.dataset) : {len(train_loader.sampler)} {len(train_loader.dataset)}")
        
        if self.is_distributed and self.use_cuda:
            # multi-machine multi-gpu case
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            # single-machine multi-gpu case or single-machine or multi-machine cpu case
            model = torch.nn.DataParallel(model)

        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        scheduler = StepLR(optimizer, step_size=3, gamma=self.args.gamma)

        for epoch in range(1, self.args.epochs + 1):
            self.train(model, train_loader, optimizer, epoch)
            if (self.rank == 0):
                # self.test(model, test_loader, epoch)
                model_prefix = f"epoch-{epoch}"
                model_path = save_model(model, self.args.model_dir, model_prefix)
                #TODO
                self.upload_model_to_s3(model_path, model_path)
            scheduler.step()
            


    def train(self, model, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, targets,_) in enumerate(train_loader, 1):
            # Print bounding boxes for debugging
            # print(f"=====[ epoch {epoch} batch {batch_idx}  data: {data}")
            # print(f"=====[ epoch {epoch} batch {batch_idx}  targets: {targets}")
            data = list(image.to(self.device) for image in data)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            print(f"=====[ epoch {epoch} batch {batch_idx} out of {len(train_loader)}  before model parse")
            output = model(data, targets)
            print(f"=====[ epoch {epoch} batch {batch_idx} output: {output}")

            total_loss = sum(output.values())
            total_loss.backward()
            print(f"=====[ epoch {epoch} batch {batch_idx} loss: {total_loss}")

            optimizer.step()

    def test(self, model, test_loader, epoch):
        model.eval()
        test_loss = 0
        correct = 0
        correct_labels = 0
        total_samples = 0
        img_pathes = []
        for batch_id, (data, targets, original_image) in enumerate(test_loader, 1):
            data = list(image.to(self.device) for image in data)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            print(f"=====[ epoch {epoch} test loop {batch_id} ")
            print(f"=====[ epoch {epoch} test loop {batch_id}  ] targets: {targets}")
            with torch.no_grad():
                prediction = model(data)
            print(f"=====[ epoch {epoch} test loop {batch_id}  ] after output: {prediction}")
            for idx, (prediction_dict, target_dict) in enumerate(zip(prediction, targets)):
                print(f"=====[ epoch {epoch} test loop {batch_id}  ] BEFORE _clean_targets: {target_dict}")
                target_dict = clean_targets(target_dict)
                print(f"=====[ epoch {epoch} test loop {batch_id} ] AFTER _clean_targets: {target_dict}")
                prediction_dict = filter_prediction(predicted_dict=prediction_dict, max_predictions=len(target_dict["labels"]))
                print(f"=====[ epoch {epoch} test loop {batch_id}  ] AFTER FILTERING prediction_dict: {prediction_dict}")

                matching_labels = (prediction_dict["labels"] == target_dict["labels"]).sum().item()
                print(f"=====[ epoch {epoch} test loop {batch_id}  ] matching_labels: {matching_labels}")

                total_samples += 1
                correct += matching_labels
                correct_labels += len(target_dict["labels"])

                # box_loss = torch.nn.functional.smooth_l1_loss(prediction_dict['boxes'], target_dict['boxes'])
                # print(f"=====[ epoch {epoch} test loop {batch_id} ] box_loss: {box_loss}")


                # Combine the losses (you can adjust the weights based on your specific task)
                # total_loss = box_loss
                # Accumulate the loss for the batch
                # test_loss += total_loss.item()
                test_loss = 0
                prediction[idx] = prediction_dict
                print(f"=====[ epoch {epoch} test loop outside {batch_id}  ] idx: {idx}")
                name_prefix = f"epoch-{epoch}-batch-{str(batch_id)}-{str(target_dict['idx'])}"
                print(f"=====[ epoch {epoch} test loop outside {batch_id}  ] name_prefix: {name_prefix}")
                img_pathes.append(save_test_img(original_image[idx], prediction[idx], name_prefix))
                print(f"===== epoch {epoch} [test loop outside {batch_id}  ] img_pathes: {img_pathes}")
        accuracy_labels = (correct / correct_labels) * 100
        accuracy_boxes = (test_loss / total_samples)
        print(f' epoch {epoch} Accuracy Labels: {accuracy_labels:.2f}%')
        print(f' epoch {epoch} Loss Boxes: {accuracy_boxes:.2f}%')
        save_to_bucket(img_pathes, self.output_bucket_name, self.s3_session)

    def upload_model_to_s3(self, model_path, model_name):
        self.s3_session.upload_file(model_path, self.output_bucket_name, model_name)



def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_fasterrcnn_detection_model(num_classes=NUMBER_OF_CLASSES)
    model = torch.nn.DataParallel(model)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print("=====[INFO] Started the job")
    print("=====[INFO] torch version: ", torch.__version__)


    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default=0.03,
        help="gamma parameter used by scheduler",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ssd",
        help="Specify the model that is going to be trained",
    )
    # # Container environment
    # print("SM_HOSTS: ",os.environ["SM_HOSTS"])

    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    # parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    parser.add_argument("--hosts", type=list, default="1")
    parser.add_argument("--current-host", type=str, default="1")
    parser.add_argument("--model-dir", type=str, default=".")
    parser.add_argument("--train-dir", type=str, default="../../train_set")
    parser.add_argument("--test-dir", type=str, default="../../train_set")
    parser.add_argument("--num-gpus", type=int, default=0)
    print("=====[INFO] Running train loop")

    main_class = Main(parser.parse_args())
    main_class.run()
