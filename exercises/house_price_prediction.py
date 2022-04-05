from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)

    df.drop_duplicates(inplace=True)
    df = df.loc[df.zipcode.notnull()]
    df = split_zipcode(df)
    df = split_date(df)
    df.drop(["id", "date", "zipcode", "sqft_living"], inplace=True, axis=1)
    df = df.loc[filters(df)]
    df = df.loc[df.sqft_above / df.floors <= df.sqft_lot]  # Another filter to apply, we need floors > 0 first.
    df["last_changed"] = np.maximum(df.yr_built, df.yr_renovated)
    return df


def split_zipcode(df: pd.DataFrame):
    zipcodes = np.fromiter(set(df.zipcode), int)
    indicators = (df.zipcode[:, None] == zipcodes).astype(np.int)
    zip_column = pd.DataFrame(indicators, columns=zipcodes)
    df = pd.concat([df, zip_column], axis=1)
    return df

def split_date(df: pd.DataFrame):
    dates = df.date
    dates = pd.concat([dates.str.slice(0, 4), dates.str.slice(4, 6)], axis=1)
    dates.columns = ['year', 'month']
    year, month = dates['year'], dates['month']
    new_df = pd.concat([df, year], axis=1)
    new_df = pd.concat([new_df, pd.get_dummies(month)], axis=1)
    return new_df

def filters(df: pd.DataFrame):
    return (df.price >= 0) & (df.floors > 0) & (df.sqft_above > 300) & (df.sqft_basement >= 0) & (df.condition <= 5) & (
            df.condition >= 1) & (df.view >= 0) & (df.view <= 4) & (df.sqft_basement <= df.sqft_lot) & \
           (df.bathrooms > 0) & (df.yr_built <= 2022) & (df.yr_renovated <= 2022)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
