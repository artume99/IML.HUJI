from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, counts = np.unique(y, return_counts=True)

        self.pi_ = counts / y.size

        mu = []

        conc_X = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)

        for k in self.classes_:
            curr_mu = conc_X[np.where(conc_X[:, conc_X.shape[1] - 1] == k)[0]].mean(axis=0)
            mu.append(curr_mu[:curr_mu.shape[0] - 1])
        self.mu_ = np.array(mu)

        vars = []
        for k in self.classes_:
            curr_vars = conc_X[np.where(conc_X[:, conc_X.shape[1] - 1] == k)[0]].var(axis=0)
            vars.append(curr_vars[:curr_vars.shape[0] - 1])
        self.vars_ = np.array(vars)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihood = self.likelihood(X)
        pred = self.classes_[np.argmax(likelihood * self.pi_, axis=1)]
        return pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = []
        from numpy.linalg import det, inv
        for k in range(self.classes_.size):
            cov = np.diag(self.vars_[k])
            mekadem = np.sqrt(det(cov) * (2 * np.pi) ** X.shape[1])
            difference = (X - self.mu_[k])
            inside = np.diag(np.exp(np.dot(np.dot(-.5 * difference, inv(cov)), difference.T)))
            likelihood = inside / mekadem
            likelihoods.append(likelihood)

        likelihoods = np.array(likelihoods).T
        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
