from weather.model import WeatherModel
from weather.dataset import WeatherDataset, Normalization, Standardization
from weather.trainer import ModelStateCheckpoint, weather_train
import torch.nn as nn
import torch.optim as optim
import torch

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    seq_len = 15
    normalize = Normalization(min=[0, 0, 0, -7.883, -5.3, -11.500, 0.000, 14.125, -4.042, 1.904, 1.775],
                              max=[1, 1, 1, 34.208, 41.000, 30.100, 10.758, 100.0, 45.575, 37.975, 36.971])
    standard = Standardization(mean=[0, 0, 0, 17.234, 22.226, 13.007, 0.184, 72.470, 20.198, 19.402, 19.407],
                               std=[1, 1, 1, 7.032, 7.896, 6.980, 0.706, 15.886, 8.339, 6.782, 6.556])
    train_wd = WeatherDataset("data/train_weather.csv",
                              "data/places.csv",
                              is_reverse=True,
                              transform=normalize,
                              seq_len=seq_len)
    test_wd = WeatherDataset("data/test_weather.csv",
                             "data/places.csv",
                             is_reverse=True,
                             transform=normalize,
                             seq_len=seq_len)
    wm = WeatherModel(3, 8, 256, 128, seq_len, 2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device =  "cpu"
    print(wm)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(wm.parameters())
    weather_train(wm,
                  train_dataset=train_wd,
                  test_dataset=test_wd,
                  criterion=criterion, optimizer=optimizer, batch_size=64,
                  device=device,
                  epochs=500,
                  model_check_point=ModelStateCheckpoint("model/weights_weather_{epoch:02d}_{val_loss:.3f}.pt",
                                                         save_best_only=False))
