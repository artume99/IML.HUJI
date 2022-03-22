import itertools

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly
import plotly.io as pio
from matplotlib import pyplot as plt

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

    plt.plot(samples, distances, marker=".")
    plt.grid(True)
    plt.axhline(0, color='r', ls='--')
    plt.title("Distance of the estimator from the real expected value \n calculated by increasing samples")
    plt.xlabel("Number of samples")
    plt.ylabel("Distance", size=15)
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = univar_guas.pdf(X)
    plt.scatter(X, pdf, s=1)
    plt.title("Distribution of the samples")
    plt.xlabel("Value of samples")
    plt.ylabel("PDF calculation", size=15)
    plt.axvline(univar_guas.mu_, color='r', ls='--')
    plt.show()


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
    likelyhoods = []
    row = []
    pre_f1 = f[0]
    max_likelihood = float('-inf')
    max_f1_f3 = ()
    for f1, f3 in itertools.product(f, f):
        if f1 != pre_f1:
            likelyhoods.append(row)
            pre_f1 = f1
            row = []
        mu = np.array([f1, 0, f3, 0])
        likelyhood = MultivariateGaussian.log_likelihood(mu, TRUE_COV, Y)
        if likelyhood > max_likelihood:
            max_likelihood = likelyhood
            max_f1_f3 = (f1, f3)
        row.append(likelyhood)
    likelyhoods.append(row)

    x, y = np.meshgrid(f, f)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, likelyhoods, cmap='RdBu')
    ax.set(title="Log likelihood of f1, f3 for \n mu = \"[f1, 0, f3, 0]\"", xlabel="f3", ylabel="f1")
    fig.colorbar(c, ax=ax)
    plt.show()

    # Question 6 - Maximum likelihood
    print(f'max likelihood achieved at {max_f1_f3}')


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
