import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def weather_compare(true_data, fake_data, column):
    min_len = min(len(true_data), len(fake_data))
    true_data = true_data[:min_len]
    fake_data = fake_data[:min_len]

    xs = pd.to_datetime(true_data[['year', 'month', 'day']])
    true_ys = true_data[column].values.reshape(-1, 1)
    fake_ys = fake_data[column].values.reshape(-1, 1)

    ys = np.concatenate((true_ys, fake_ys), axis=1)

    plt.figure(figsize=(15, 5))
    plt.plot(xs, ys)
    plt.legend(['true', 'fake'])
    # plt.savefig("../assets/weather.png")
    plt.show()


def weather_draw(data, transform=None, condition=None, columns=None, reverse=False):
    if columns is None:
        columns = ['24_hours_avg_tem', '24_hours_max_tem',
                   '24_hours_min_tem', '24_hours_avg_pre',
                   '24_hours_avg_rhu', '24_hours_avg_gst',
                   '24_hours_avg_gst_15cm', '24_hours_avg_gst_20cm']
    if type(data) == str:
        weather_df = pd.read_csv(data)
    else:
        weather_df = data

    if condition is not None:
        weather_df = weather_df[condition(weather_df)]

    if transform is not None:
        weather_df[columns] = transform(weather_df[columns])

    xs = np.arange(len(weather_df))
    xs = pd.to_datetime(weather_df[['year', 'month', 'day']])
    if reverse:
        xs = xs[::-1]
        ys = weather_df[columns][::-1].values
    else:
        ys = weather_df[columns].values

    plt.figure(figsize=(15, 5))
    plt.plot(xs, ys)
    plt.legend(columns)
    # plt.savefig("../assets/weather.png")
    plt.show()
