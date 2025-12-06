import os
import time
import copy
import torch
import torch.nn as nn
from train.training_utils import set_optimizer, set_scheduler
from train.upload.model import saveModel, getPretrainedModel, AnyModel
from train.upload.datasets import getCOCODataset
from utils import setSeed, selectDevice

learning_rate = float(os.environ['LEARNING_RATE'])
weight_decay = float(os.environ['WEIGHT_DECAY'])
epoch_count = int(os.environ["EPOCHS"])

class Trainer:
    def __init__(self):
        setSeed()
        self.device = selectDevice()
        out = getCOCODataset()
        if len(out) == 4:
            self.train_loader, self.valid_loader, self.test_loader, self.num_classes = out
        else:  # wsteczna kompatybilność
            self.train_loader, self.valid_loader, self.num_classes = out
            self.test_loader = None
        self.model = getPretrainedModel(self.num_classes) # Faster R-CNN z odchudzeniem VRAM
        self.model = self.model.to(self.device)

        self.optimizer = set_optimizer(self.model, learning_rate, weight_decay)
        self.scheduler = set_scheduler(self.optimizer, T_max=epoch_count, eta_min=learning_rate * 0.01)

        # Nowe API GradScaler
        self.scaler = torch.amp.GradScaler(device=self.device)

        self.train_losses = []
        self.val_losses = []


    def _move_to_device(self, images, targets):
        images = [img.to(self.device, non_blocking=True) for img in images]
        targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
        return images, targets

    def _forward(self) -> float:
        self.model.train()
        running_loss = 0.0
        n = 0

        for images, targets in self.train_loader:
            images, targets = self._move_to_device(images, targets)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=(self.scaler is not None)):
                loss_dict = self.model(images, targets)  # dict strat
                loss = sum(loss_dict.values())

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            bs = len(images)
            running_loss += loss.item() * bs
            n += bs

            # Uwaga: empty_cache() spowalnia trening, usunięte dla wydajności

        return running_loss / max(n, 1)

    @torch.no_grad()
    def _evaluate(self) -> float:
        was_training = self.model.training
        # Tymczasowo train(), aby otrzymać dict strat
        self.model.train()

        # Ustaw BN w eval na czas walidacji
        bn_modules = []
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if m.training:
                    bn_modules.append(m)
                    m.eval()

        running_loss = 0.0
        n = 0

        for images, targets in self.valid_loader:
            images, targets = self._move_to_device(images, targets)
            # safety check – pomiń batch, jeśli pojawiły się degenerate boxy
            bad = False
            for t in targets:
                if t["boxes"].numel():
                    x1y1x2y2 = t["boxes"]
                    if (x1y1x2y2[:, 2] <= x1y1x2y2[:, 0]).any() or (x1y1x2y2[:, 3] <= x1y1x2y2[:, 1]).any():
                        bad = True
                        break
            if bad:
                continue

            with torch.autocast(device_type=self.device.type, enabled=(self.scaler is not None)):
                loss_dict = self.model(images, targets)  # dict strat (dostępny w trybie train)
                if isinstance(loss_dict, list):  # asekuracja
                    continue
                loss = sum(loss_dict.values())

            bs = len(images)
            running_loss += loss.item() * bs
            n += bs

        # Przywróć stan BN i tryb modelu
        for m in bn_modules:
            m.train()
        if not was_training:
            self.model.eval()

        return running_loss / max(n, 1)

    def run(self):
        best_val_loss = float("inf")
        best_wts = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0
        start_time = time.time()

        for epoch in range(epoch_count):
            print(f"Epoch {epoch + 1}/{epoch_count}")

            train_loss = self._forward()
            val_loss = self._evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            print(f"Training: loss: {train_loss:.4f} | val: loss: {val_loss:.4f} | "
                  f"lr: {self.optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= int(os.environ["PATIENCE"]):
                print(f"No improvement after {epochs_no_improve} epochs. Saving best model!")
                break

        elapsed_time = time.time() - start_time
        print(f"\nKoniec treningu. Najlepszy val loss = {best_val_loss:.4f}. Czas: {elapsed_time / 60:.1f} min.")

        saveModel(
            self.model, best_wts, self.device,
            best_val_loss=best_val_loss,
            train_losses=self.train_losses, val_losses=self.val_losses
        )
