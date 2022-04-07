import matplotlib.pyplot as plt

import IMLearn.learners.regressors.linear_regression as lin
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'], dayfirst=True)
    df = df.dropna()
    df = df.loc[(df.Temp >= -20) & (df.Temp <= 50)]
    df.index = np.arange(len(df))
    dayOfYear = [date.dayofyear for date in df['Date']]
    df["DayOfYear"] = dayOfYear

    return df


def reduce_to_country(country: str, df: pd.DataFrame):
    new_df = df[df["Country"] == country]
    cmap = plt.get_cmap('jet', 20)
    cmap.set_under('gray')
    cax = plt.scatter(new_df["DayOfYear"], new_df["Temp"], c=new_df["Year"], s=5, cmap=cmap)
    plt.title("Temperature in Israel over the Years")
    plt.xlabel("Day Of The Year")
    plt.ylabel("Temperature in c°")
    plt.colorbar(cax, extend='min')
    plt.show()

    std_of_month = new_df.groupby(["Month", "Day"]).agg({'Temp': "std"})
    std_of_month = std_of_month.groupby("Month").mean().to_numpy()

    samples = np.arange(1, 13)  # number of months

    plt.bar(samples, std_of_month.flatten())
    plt.xticks(np.arange(min(samples), max(samples) + 1))
    plt.xlabel("Month")
    plt.ylabel("Std of the Temperature")
    plt.title("Standard deviation of the temperature per month")
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    reduce_to_country("Israel", df)

    # Question 3 - Exploring differences between countries
    countries = df.groupby(["Country"])
    samples = np.arange(1, 13)
    std_groups = np.array([countries.get_group(country).groupby(["Month", "Day"]).agg({'Temp': "std"}).groupby(
        "Month").mean().to_numpy() for country in countries.groups])
    mean_groups = np.array([countries.get_group(country).groupby(["Month", "Day"]).agg({'Temp': "mean"}).groupby(
        "Month").mean().to_numpy() for country in countries.groups])
    colors = ["navy", "black", "forestgreen", "orange"]
    i = 0
    for mean, std, color in zip(mean_groups, std_groups, colors):
        plt.plot(samples, mean.flatten(), color=color, label=list(countries.groups.keys())[i])
        plt.fill_between(np.arange(1, 13), mean.flatten() + std.flatten(), mean.flatten() - std.flatten(), alpha=0.2,
                         color=color)
        i += 1
    plt.legend(loc="upper left")
    plt.xlabel("Month")
    plt.ylabel("Mean of temperature")
    plt.title("Mean of temperature as a function of Month")
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    israel = df.loc[df['Country'] == "Israel"]
    israel_temp = israel.pop("Temp")
    israel_day_of_year = pd.DataFrame({"DayOfYear": israel.pop("DayOfYear")})
    x_train, y_train, x_test, y_test = split_train_test(israel_day_of_year, israel_temp)

    losses = []
    samples = np.arange(1, 11)
    for k in samples:
        pl = PolynomialFitting(k)
        pl.fit(x_train.to_numpy(), y_train.to_numpy())
        loss = pl.loss(x_test.to_numpy(), y_test.to_numpy())
        losses.append(loss)
        print(f'Dimension {k} with loss {loss}')
    plt.bar(samples, losses)
    plt.xticks(np.arange(min(samples), max(samples) + 1))
    plt.xlabel("K - Dimension")
    plt.ylabel("Loss over the training set")
    plt.title("Loss of the training set by increasing dimension K")
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    min_k = np.argmin(losses)
    model = PolynomialFitting(min_k)
    model.fit(israel_day_of_year.to_numpy().astype(float), israel_temp.to_numpy())
    losses = []
    countries = df['Country'].unique()
    for country in countries:
        country = df.loc[df['Country'] == country]
        country_temp = country.pop("Temp")
        country_day_of_year = pd.DataFrame({"DayOfYear": country.pop("DayOfYear")})
        loss = model.loss(country_day_of_year.to_numpy().astype(float), country_temp.to_numpy())
        losses.append(loss)

    plt.bar(countries, losses)
    plt.xlabel("K")
    plt.ylabel("Loss")
    plt.title("Loss as a function of K")
    plt.show()
