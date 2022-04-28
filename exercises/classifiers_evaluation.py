from collections import namedtuple

import matplotlib.axes
import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import matplotlib.pyplot as plt


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = load_dataset(f'../datasets/{f}')

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit._loss(data[0], data[1]))

        perceptron = Perceptron(callback=callback)
        perceptron.fit(data[0], data[1])

        # region Plot figure of loss as function of fitting iteration
        plt.plot(losses)
        plt.title(f'Perceptron misclassification loss per Iteration \n for {n}')
        plt.xlabel("Iterations")
        plt.ylabel("Missclassification Loss", size=15)
        plt.show()
        # endregion


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = load_dataset(f'../datasets/{f}')
        x = data[0]
        y = data[1]

        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        lda.fit(x, y)
        gnb.fit(x, y)
        pred_lda = lda.predict(x)
        pred_gnb = gnb.predict(x)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        accuracy_lda = accuracy(y, pred_lda)
        accuracy_gnb = accuracy(y, pred_gnb)

        lda_title = f"LDA classification with \n accuracy = {accuracy_lda}"
        gnb_title = f'GaussianNaiveBayes classification with \n accuracy = {accuracy_gnb}'
        fig = make_subplots(rows=1, cols=2, subplot_titles=(gnb_title, lda_title))
        fig.update_layout(title_text="Classification by different models")
        fig.update_xaxes(title_text="feature1", row=1, col=1)
        fig.update_xaxes(title_text="feature1", row=1, col=2)
        fig.update_yaxes(title_text="feature2", row=1, col=1)
        fig.update_yaxes(title_text="feature2", row=1, col=2)

        # Add traces for data-points setting symbols and colors
        fig.add_trace(
            go.Scatter(mode='markers', x=x[:, 0], y=x[:, 1],
                       marker=dict(color=pred_gnb, symbol=y, size=10)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(mode='markers', x=x[:, 0], y=x[:, 1],
                       marker=dict(color=pred_lda, symbol=y, size=10)),
            row=1, col=2
        )

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(mode='markers', x=gnb.mu_[:, 0], y=gnb.mu_[:, 1],
                       marker=dict(color='black', symbol='x')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(mode='markers', x=lda.mu_[:, 0], y=lda.mu_[:, 1],
                       marker=dict(color='black', symbol='x')),
            row=1, col=2
        )

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_traces([get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i]))
                        for i in range(gnb.classes_.size)], rows=1, cols=1)

        fig.add_traces([get_ellipse(lda.mu_[i], lda.cov_)
                        for i in range(lda.classes_.size)], rows=1, cols=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)

    X = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
    y = np.array([0, 0, 1, 1, 1, 1])
    p = GaussianNaiveBayes()
    p.fit(X, y)

    run_perceptron()
    # compare_gaussian_classifiers()
