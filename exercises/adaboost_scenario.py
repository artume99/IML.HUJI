import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # region q1
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    boost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_loss, test_loss = [], []
    for t in range(1, n_learners + 1):
        train_loss.append(boost.partial_loss(train_X, train_y, t))
        test_loss.append(boost.partial_loss(test_X, test_y, t))

    plt.plot(train_loss, label="Train loss")
    plt.plot(test_loss, label="Test loss")
    plt.grid()
    plt.title("training- and test errors as a function of the number of fitted learners.")
    plt.xlabel("number of iterations")
    plt.ylabel("error rate")
    plt.legend(loc="upper right")
    plt.show()
    # endregion

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{m} Iterations" for m in T])
    fig.update_layout(title=f"Decision Boundaries with a noise of {noise}",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: boost.partial_predict(T=t, X=X), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.show()
    fig.write_image(f"q2_noise_{noise}.png")

    # Question 3: Decision surface of best performing ensemble
    scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                         marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                     line=dict(color="black", width=1)))
    min_iterations = np.argmin(test_loss) + 1
    y_pred = boost.partial_predict(test_X, min_iterations)
    acc = accuracy(test_y, y_pred)
    fig = go.Figure(
        [decision_surface(lambda X: boost.partial_predict(X, min_iterations), lims[0], lims[1], showscale=False),
         scatter])

    fig.update_layout(
        title_text=f"Ensemble achieved the lowest test error with \n{min_iterations} classifiers and {acc} accuracy")
    fig.show()
    fig.write_image(f"q3_noise_{noise}.png")

    # Question 4: Decision surface with weighted samples
    scatter = go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                         mode="markers", showlegend=False,
                         marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],
                                     line=dict(color="black", width=1),
                                     size=(boost.D_ / np.max(boost.D_)) * 5))
    fig = go.Figure([decision_surface(boost.predict, lims[0], lims[1], showscale=False), scatter])
    fig.update_layout(
        title=f"Decision surface with weighted samples using decision stump with {noise} noise.",
        margin=dict(t=100))
    fig.show()
    fig.write_image(f"q4_noise_{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
