import torch.nn as nn
from torch.utils.data import DataLoader

from weather.dataset import WeatherDataset
import torch
import numpy as np
from time import time


class ModelStateCheckpoint:

    def __init__(self, filepath, save_best_only=False, mode='min', monitor='val_loss'):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        if mode == 'min':
            self.best = np.Inf
            self.monitor_op = np.less
        else:
            self.best = - np.Inf
            self.monitor_op = np.greater

    def set_model(self, model: nn.Module):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if self.monitor not in logs:
            raise RuntimeError(f"{self.monitor} not in logs")
        current = logs.get(self.monitor)
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                torch.save(self.model.state_dict(), filepath)
        else:
            torch.save(self.model.state_dict(), filepath)


def weather_train(net: nn.Module,
                  train_dataset: WeatherDataset,
                  criterion,
                  optimizer,
                  test_dataset: WeatherDataset = None,
                  batch_size=4,
                  epochs=120,
                  log_steps=10,
                  device="cpu",
                  model_check_point: ModelStateCheckpoint = None):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    net.to(device)

    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if model_check_point is not None:
        model_check_point.set_model(net)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_loss = 0.0
        logs = {}
        time_since_last = time()
        for i, data in enumerate(train_dataloader, 0):
            enc_inputs, target, dec_inputs = data.values()
            enc_inputs, target, dec_inputs = enc_inputs.float(), target.float(), dec_inputs.float()
            enc_inputs, target, dec_inputs = enc_inputs.to(device), target.to(device), dec_inputs.to(device)

            optimizer.zero_grad()

            outputs, _, _ = net(enc_inputs, dec_inputs)
            loss = criterion(outputs, target)
            loss.backward(torch.ones_like(loss.data))
            optimizer.step()

            running_loss += loss.item()

            if i % log_steps == log_steps - 1:
                train_loss = running_loss / log_steps
                print(f"[{epoch + 1:3d} {i + 1: 5d}] loss: {running_loss / log_steps:.5f} ")
                running_loss = 0.0
        time_curr = time()

        logs['loss'] = train_loss

        if test_dataset is not None:
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in test_dataloader:
                    enc_inputs, target, dec_inputs = data.values()
                    enc_inputs, target, dec_inputs = enc_inputs.float().to(device), target.float().to(device), \
                                                     dec_inputs.float().to(device)
                    outputs, _, _ = net(enc_inputs, dec_inputs)
                    loss = criterion(outputs, target)
                    val_loss += loss.item()

            val_loss = val_loss / len(test_dataloader)
            logs['val_loss'] = val_loss

        print(
            f"[{epoch + 1:3d}] - {(time_curr - time_since_last):.0f}s"
            f" - {((time_curr - time_since_last) * 1000 // len(train_dataloader)):.0f}ms/step"
            f" - loss: {train_loss:.5f}"
            + (f" - val_loss: {val_loss:.5f}" if test_dataset is not None else ""))

        if model_check_point is not None:
            model_check_point.on_epoch_end(epoch, logs)
