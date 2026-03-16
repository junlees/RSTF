"""
train_cnc.py — Training script for CNC TransformerAnomalyDetector
─────────────────────────────────────────────────────────────────
Usage:
    python train_cnc.py -c config_cnc_transformer.json
    python train_cnc.py -c config_cnc_transformer.json --lr 0.0001
    python train_cnc.py -r saved/models/CNC_TransformerAutoencoder/XXXX/checkpoint-epoch5.pth
"""

import argparse
import collections
import torch
import numpy as np

import data_loader.cnc_data_loaders as module_data
import model.loss    as module_loss
import model.metric  as module_metric
import model.transformer_model as module_arch
from parse_config import ConfigParser
from trainer.anomaly_trainer import AnomalyTrainer
from utils import prepare_device

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False   # benchmark=True와 호환
torch.backends.cudnn.benchmark     = True    # 입력 크기 고정 시 최적 커널 자동 선택
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # ── Data ─────────────────────────────────────────────────────────────
    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    # ── Model ────────────────────────────────────────────────────────────
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # ── Device ───────────────────────────────────────────────────────────
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = torch.compile(model)   # PyTorch 2.0+ 커널 컴파일 (첫 배치에서 ~30초 소요)

    # ── Loss & Metrics ───────────────────────────────────────────────────
    # The AnomalyTrainer constructs CombinedAnomalyLoss internally;
    # here we still pass the named loss for logging / config consistency.
    criterion = getattr(module_loss, config["loss"])
    metrics   = [getattr(module_metric, m) for m in config["metrics"]]

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer        = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler     = config.init_obj(
        "lr_scheduler", torch.optim.lr_scheduler, optimizer
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = AnomalyTrainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="CNC Transformer Autoencoder — Train")
    args.add_argument("-c", "--config", default=None, type=str,
                      help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="indices of GPUs to enable (default: all)")

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float,
                   target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"],    type=int,
                   target="data_loader;args;batch_size"),
        CustomArgs(["--alpha"],                 type=float,
                   target="trainer;loss_alpha"),
        CustomArgs(["--machines"],              type=str,
                   target="data_loader;args;machines"),
        CustomArgs(["--ops", "--operations"],   type=str,
                   target="data_loader;args;operations"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
