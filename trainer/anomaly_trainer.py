"""
Trainer for TransformerAnomalyDetector.

Key differences from the base template Trainer
────────────────────────────────────────────────
1. Model outputs (logits, reconstruction) — both are used in the loss.
2. Loss = alpha * CrossEntropy(logits, labels) + (1-alpha) * MSE(reconstruction, input)
3. Separate metric tracking for classification and reconstruction.
4. Logs reconstruction error as an extra diagnostic metric.
"""

import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.loss import CombinedAnomalyLoss


class AnomalyTrainer(BaseTrainer):
    """
    Trainer class for Transformer Autoencoder anomaly detection.

    config additions (inside "trainer" block)
    ──────────────────────────────────────────
    "loss_alpha": 0.6   ← weight for CE loss  (1-alpha for reconstruction loss)
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader

        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = max(1, int(np.sqrt(data_loader.batch_size)))

        # Combined loss (CE + reconstruction)
        loss_alpha = config["trainer"].get("loss_alpha", 0.6)
        self.combined_loss = CombinedAnomalyLoss(alpha=loss_alpha).to(device)

        # Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda")

        # Metric trackers — track classification + reconstruction metrics
        metric_names = [m.__name__ for m in self.metric_ftns]
        extra = ["ce_loss", "recon_loss"]
        self.train_metrics = MetricTracker(
            "loss", *extra, *metric_names, writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *extra, *metric_names, writer=self.writer
        )

    # ── Training epoch ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data   = data.to(self.device)    # (B, W, F)
            target = target.to(self.device)  # (B,)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                output = self.model(data)    # (logits, reconstruction)
                loss, ce_loss, recon_loss = self.combined_loss(output, target, data)

            self.scaler.scale(loss).backward()
            # Gradient clipping for stable Transformer training
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            step = (epoch - 1) * self.len_epoch + batch_idx
            self.writer.set_step(step)
            self.train_metrics.update("loss",       loss.item())
            self.train_metrics.update("ce_loss",    ce_loss.item())
            self.train_metrics.update("recon_loss", recon_loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.4f} CE: {:.4f} Recon: {:.4f}".format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item(),
                        ce_loss.item(),
                        recon_loss.item(),
                    )
                )

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    # ── Validation epoch ──────────────────────────────────────────────────

    def _valid_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data   = data.to(self.device)
                target = target.to(self.device)

                with torch.amp.autocast("cuda"):
                    output = self.model(data)
                    loss, ce_loss, recon_loss = self.combined_loss(output, target, data)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx,
                    "valid",
                )
                self.valid_metrics.update("loss",       loss.item())
                self.valid_metrics.update("ce_loss",    ce_loss.item())
                self.valid_metrics.update("recon_loss", recon_loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        # Log model parameter histograms
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")

        return self.valid_metrics.result()

    # ── Progress helper ───────────────────────────────────────────────────

    def _progress(self, batch_idx: int) -> str:
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total   = self.data_loader.n_samples
        else:
            current = batch_idx
            total   = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
