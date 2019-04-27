from weather.predictor import weather_predict
from weather.model import WeatherModel
from weather.dataset import PreWeatherDataset, Normalization, Standardization
from drawing.weather_drawing import weather_draw, weather_compare
import torch
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    max_seq = 15
    normalize = Normalization(min=[0, 0, 0, -7.883, -5.3, -11.500, 0.000, 14.125, -4.042, 1.904, 1.775],
                              max=[1, 1, 1, 34.208, 41.000, 30.100, 10.758, 100.0, 45.575, 37.975, 36.971])
    standard = Standardization(mean=[0, 0, 0, 17.234, 22.226, 13.007, 0.184, 72.470, 20.198, 19.402, 19.407],
                               std=[1, 1, 1, 7.032, 7.896, 6.980, 0.706, 15.886, 8.339, 6.782, 6.556])
    wd = PreWeatherDataset("data/forecast_test.csv",
                           "data/places.csv",
                           transform=normalize,
                           seq_len=max_seq,
                           is_reverse=False)
    model = WeatherModel(3, 8, 256, 128, max_seq, 2, dropout=0.2)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("model/weights_weather_07_0.001.pt", map_location=device))

    weather = weather_predict(model, wd, num_future=120)

    weather_draw(weather)

    true_weather = pd.read_csv("data/weather.csv")
    true_weather = true_weather[(true_weather['county'] == '仁和') & (true_weather['year'] == 2017)
                                & (true_weather['month'] >= 5)][::-1]
    true_weather = true_weather[:len(weather)]
    weather_draw(true_weather)

    weather_compare(true_weather, weather, '24_hours_avg_tem', save_path='data/weather.png')
