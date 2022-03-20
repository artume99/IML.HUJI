import itertools

import matplotlib.pyplot as plt

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly
import plotly.io as pio
from matplotlib import pyplot as pt

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    SAMPLE_NUM = 1000
    TRUE_EXPECTED = 10
    TRUE_SIGMA = 1
    univar_guas = UnivariateGaussian()
    X = fit_model(TRUE_EXPECTED, TRUE_SIGMA, SAMPLE_NUM, univar_guas)
    print(univar_guas)

    # Question 2 - Empirically showing sample mean is consistent
    exception_estimator = []
    samples = []
    temp_univar_guas = UnivariateGaussian()
    for m in range(10, SAMPLE_NUM, 10):
        fit_model(TRUE_EXPECTED, TRUE_SIGMA, m, temp_univar_guas)
        exception_estimator.append(temp_univar_guas.mu_)
        samples.append(m)
    exception_estimator = np.array(exception_estimator)
    distances = np.abs(exception_estimator - TRUE_EXPECTED)

    pt.plot(samples, distances, marker=".")
    pt.grid(True)
    pt.axhline(0, color='r', ls='--')
    pt.title("Distance of the estimator from the real expected value \n calculated by increasing samples")
    pt.xlabel("Number of samples")
    pt.ylabel("Distance", size=15)
    pt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = univar_guas.pdf(X)
    pt.scatter(X, pdf, s=1)
    pt.title("Distribution of the samples")
    pt.xlabel("Value of samples")
    pt.ylabel("PDF calculation", size=15)
    pt.axvline(univar_guas.mu_, color='r', ls='--')
    pt.show()


def fit_model(mu, sigma, samples, gaus) -> np.ndarray:
    X = np.random.normal(mu, sigma, samples)
    gaus.fit(X)
    return X


def fit_multi_model(mu, cov, samples, gaus) -> np.ndarray:
    X = np.random.multivariate_normal(mu, cov, samples)
    gaus.fit(X)
    return X


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    SAMPLE_NUM = 1000
    TRUE_EXPECTED = np.array([0, 0, 4, 0])
    TRUE_COV = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    multivar_gaus = MultivariateGaussian()
    Y = fit_multi_model(TRUE_EXPECTED, TRUE_COV, SAMPLE_NUM, multivar_gaus)
    print(multivar_gaus)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    for f1, f3 in itertools.product(f, f):
        mu = np.array([f1, 0, f3, 0])
        likelyhood = MultivariateGaussian.log_likelihood(mu, TRUE_COV, Y)


    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    # x = np.array([[150, 45], [170, 74], [184, 79]])
    # y = np.array([168, 66])

    # z = np.array([[1, 3], [3, 5], [5, 7]])
    # for_sum = np.ones((1, z.shape[0]))
    # mean = x.mean(axis=0)
    # central_matrix = x - mean[None, :]
    # var = np.dot(central_matrix.T, central_matrix) / (x.shape[0] - 1)
    # kl = MultivariateGaussian()
    # Y = np.random.multivariate_normal([0, 0, 4, 0], [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]],
    #                                   1000)
    # print(Y.shape)
    # for i in range(200 ** 2):
    #     MultivariateGaussian.log_likelihood(np.array([0, 0, 4, 0]),
    #                                         np.array([[1, 0.2, 0, 0.5], [0.1, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]]),
    #                                         Y)
    #     print(i)
    test_multivariate_gaussian()
