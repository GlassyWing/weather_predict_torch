import pandas as pd

if __name__ == '__main__':
    weather = pd.read_csv("../data/weather.csv")
    train_weather = weather[weather['year'] != 2015]
    test_weather = weather[weather['year'] == 2015]

    train_weather.to_csv("../data/train_weather.csv", index=False)
    test_weather.to_csv("../data/test_weather.csv", index=False)
