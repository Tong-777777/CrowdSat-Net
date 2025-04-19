import torch.nn
from datasets.transforms import *
import albumentations as A
from datasets.crowd_dataset import CrowdDataset
import os
import time
import argparse
import datetime
import yaml
import pprint
import pandas as pd
from utils.loss import FocalLoss
from utils.seed import set_seed
from torch import Tensor
from torch.nn import CrossEntropyLoss
from model import LossWrapper, CrowdSatNet
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from utils.logger import *
from utils.averager import *
from utils.metrics import *
from typing import Optional, Union
from utils.lmds import LMDS
import warnings
from colorama import Fore, Style, init

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training CrowdSat-Net', add_help=False)
    parser.add_argument("--config", default="./configs/crowdsat.yaml", help="Path to the config file (yaml type)")

    return parser.parse_args()

def main(args):

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    ###########################
    # 1. seed and output settings
    ###########################
    set_seed(cfg["seed"])

    device = torch.device(cfg["device_name"])

    work_dir = cfg["work_dir"]
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    logger_path = os.path.join(cfg["work_dir"], cfg["log_path"])
    os.makedirs(logger_path, exist_ok=True)
    logger, curr_timestr = setup_default_logging("global", logger_path)
    logger.info("{}".format(pprint.pformat(cfg)))
    curr_timestr = curr_timestr.replace(':', '_')
    csv_path = os.path.join(logger_path, "detection_{}_stat.csv".format(curr_timestr))


    ###########################
    # 2. model and loss function preparation
    ###########################
    model = CrowdSatNet(num_classes = cfg["datasets"]["num_classes"])
    model.to(device)

    losses = [{"loss": FocalLoss(reduction="mean"), "name": "first_order_loss"},
              {"loss": FocalLoss(reduction="mean"), "name": "second_order_loss"}]

    model = LossWrapper(model, losses=losses)

    optimizer = AdamW(params=model.parameters(), lr=float(cfg["training_settings"]["lr"]), weight_decay=float(cfg["training_settings"]["weight_decay"]))

    ###########################
    # 3. datasets preparation
    ###########################
    train_set = CrowdDataset(
        data_root = cfg["train"]["data_root"],
        train = True,
        albu_transforms = [
            # A.VerticalFlip(p = cfg["train"]["albu_transforms"]["HorizontalFlip"]["p"]),
            A.Normalize(p = cfg["train"]["albu_transforms"]["Normalize"]["p"])
        ],
        end_transforms = [MultiTransformsWrapper([
            FIDT(num_classes = cfg["train"]["end_transforms"]["MultiTransformsWrapper"]["FIDT"]["num_classes"], down_ratio = cfg["train"]["end_transforms"]["MultiTransformsWrapper"]["FIDT"]["down_ratio"]),
            PointsToMask(radius = cfg["train"]["end_transforms"]["MultiTransformsWrapper"]["PointsToMask"]["radius"], num_classes = cfg["train"]["end_transforms"]["MultiTransformsWrapper"]["PointsToMask"]["num_classes"], squeeze = cfg["train"]["end_transforms"]["MultiTransformsWrapper"]["PointsToMask"]["squeeze"], down_ratio = cfg["train"]["end_transforms"]["MultiTransformsWrapper"]["PointsToMask"]["down_ratio"])
        ])])

    val_set = CrowdDataset(
        data_root = cfg["train"]["data_root"],
        train = False,
        albu_transforms = [A.Normalize(p=cfg["train"]["validate"]["albu_transforms"]["Normalize"]["p"])],
        end_transforms = [DownSample(down_ratio=cfg["train"]["validate"]["end_transforms"]["DownSample"]["down_ratio"], crowd_type=cfg["train"]["validate"]["end_transforms"]["DownSample"]["crowd_type"])]
    )

    train_dataloader = DataLoader(
        dataset = train_set,
        batch_size = cfg["training_settings"]["batch_size"],
        shuffle = True
    )

    val_dataloader = DataLoader(
        dataset = val_set,
        batch_size = cfg["val_settings"]["batch_size"],
        shuffle = False
    )

    ###########################
    # 4. training loop
    ###########################

    metrics = PointsMetrics(radius=2, num_classes=cfg['datasets']['num_classes'])

    checkpoints = cfg["val_settings"]["checkpoint"]
    select = cfg["val_settings"]["select_mode"]
    validate_on = cfg["val_settings"]["validate_on"]

    assert checkpoints in ['best', 'all', 'latest']
    assert select in ['min', 'max']

    print_freq = cfg["training_settings"]["print_freq"]

    last_epoch = 0
    best_epoch = -1

    if select == 'min':
        best_val = float('inf')
    elif select == 'max':
        best_val = 0

    logger.info('-------------------------- start training --------------------------')
    for epoch in range(last_epoch, cfg["training_settings"]["epochs"]):

        loss = train(
            model,
            train_dataloader,
            optimizer,
            epoch,
            device,
            logger,
            print_freq,
            cfg
        )

        tmp_results = validate(
            model,
            val_dataloader,
            epoch,
            metrics,
            cfg
        )

        is_best = False

        if select == 'min':
            if tmp_results[validate_on] < best_val:
                best_val = tmp_results[validate_on]
                best_epoch = epoch
                is_best = True

        elif select == 'max':
            if tmp_results[validate_on] > best_val:
                best_val = tmp_results[validate_on]
                best_epoch = epoch
                is_best = True

        # save checkpoints
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_val': best_val
        }

        if is_best:
            outpath = os.path.join(work_dir, 'best_model.pth')
            torch.save(state, outpath)
        else:
            outpath = os.path.join(work_dir, 'latest_model.pth')
            torch.save(state, outpath)

        tmp_results['best_val'] = best_val
        tmp_results['best_epoch'] = best_epoch

        data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch + 1))
        data_frame.to_csv(csv_path, mode='a', header=None, index_label='epoch')

        logger.info(
            f"{Fore.CYAN}<<Test>>{Style.RESET_ALL} - "
            f"Epoch: {Fore.CYAN}{epoch}{Style.RESET_ALL}.  "
            f"{Fore.BLUE}{validate_on}{Style.RESET_ALL}: {Fore.GREEN}{tmp_results[validate_on]:.4f}{Style.RESET_ALL}.  "
            f"{Fore.BLUE}Best-Val:{Style.RESET_ALL}{Fore.RED}{best_val:.4f}{Style.RESET_ALL}  "
            f"{Fore.BLUE}Best-Epoch:{Style.RESET_ALL}{Fore.RED}{best_epoch}{Style.RESET_ALL}"
        )

def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        device: torch.device,
        logger: logging.Logger,
        print_freq: int,
        cfg: dict
        ) -> float:
    """
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer used to update the model parameters.
        epoch (int): Current epoch number during the training process.
        device (torch.device): Device on which computations are performed (e.g., 'cpu' or 'cuda').
        logger (logging.Logger): Logger instance for recording training progress and details.
        cfg (dict): Configuration dictionary containing hyperparameters and other settings.

    Returns:
        float: The sum training loss for the current epoch.
    """

    # metric indicators
    first_order_loss = AverageMeter(20)
    second_order_loss = AverageMeter(20)
    losses = AverageMeter(20)
    batch_times = AverageMeter(20)

    # print freq 8 times for a epoch
    freq = len(train_dataloader) // print_freq
    print_freq_lst = [i * freq for i in range(1, 8)]
    print_freq_lst.append(len(train_dataloader) - 1)

    batch_start = time.time()

    model.train()
    for step, (images, targets) in enumerate(train_dataloader):

        images = images.to(device)

        if isinstance(targets, (list, tuple)):
            targets = [tar.to(device) for tar in targets]
        else:
            targets = targets.to(device)

        loss_dict = model(images, targets)
        first_order_loss.update(loss_dict['first_order_loss'].item())
        second_order_loss.update(loss_dict['second_order_loss'].item())

        loss = sum(loss for loss in loss_dict.values())
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in print_freq_lst:
            logger.info(
                "Epoch/Iter [{}:{:3}/{:3}].  "
                "First:{first_order_loss.val:.3f}({first_order_loss.avg:.3f})  "
                "Second:{second_order_loss.val:.3f}({second_order_loss.avg:.3f})  "
                "Loss:{losses.val:.3f}({losses.avg:.3f})  ".format(
                    cfg["training_settings"]["epochs"], epoch, step,
                    first_order_loss=first_order_loss,
                    second_order_loss=second_order_loss,
                    losses=losses,
                )
            )

    out = losses.avg

    batch_end = time.time()
    batch_times.update(batch_end-batch_start)

    logger.info(
        "Epoch [{}].  "
        "First:{first_order_loss.val:.3f}({first_order_loss.avg:.3f})  "
        "Second:{second_order_loss.val:.3f}({second_order_loss.avg:.3f})  "
        "Loss:{losses.val:.3f}({losses.avg:.3f})  "
        "Time:{batch_times.avg:.2f}  ".format(
            cfg["training_settings"]["epochs"],
            first_order_loss=first_order_loss,
            second_order_loss=second_order_loss,
            losses=losses,
            batch_times=batch_times
        ))

    return out

@torch.no_grad
def validate(
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        epoch: int,
        metrics: object,
        cfg: dict
        ) -> Union[float, torch.Tensor]:
    """
       Validate the model on the validation dataset.

       Args:
           model (torch.nn.Module): The neural network model to be validated.
           val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
           epoch (int): Current epoch number during validation.
           device (torch.device): Device on which computations are performed (e.g., 'cpu' or 'cuda').

       Returns:
           Optional[float, torch.Tensor]:
               - `float`: The average validation loss for the epoch.
               - `torch.Tensor`: The prediction matrix containing model predictions for the validation dataset.
       """

    metrics.flush()

    iter_metrics = metrics.copy()

    model.eval()

    for step, (images, targets) in enumerate(val_dataloader):

        images = images.cuda()
        output, _ = model(images)

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()

        gt = dict(
            loc=gt_coords,
            labels=gt_labels
        )

        lmds = LMDS(kernel_size=cfg["val_settings"]["lmds_kwargs"]["kernel_size"], adapt_ts=cfg["val_settings"]["lmds_kwargs"]["adapt_ts"])

        counts, locs, labels, scores = lmds(output)

        preds = dict(
            loc=locs[0],
            labels=labels[0],
            scores=scores[0],
        )

        iter_metrics.feed(**dict(gt = gt, preds = preds))
        iter_metrics.aggregate()

        iter_metrics.flush()
        metrics.feed(**dict(gt=gt, preds=preds))

    mAP = np.mean([metrics.ap(c) for c in range(1, metrics.num_classes)]).item()

    metrics.aggregate()

    recall = metrics.recall()
    precision = metrics.precision()
    f1_score = metrics.fbeta_score()
    accuracy = metrics.accuracy()

    tmp_results = {
        'epoch': epoch,
        'f1_score': f1_score,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        "mAP": mAP
    }

    return tmp_results


if __name__ == '__main__':
    args = get_args_parser()

    main(args)

