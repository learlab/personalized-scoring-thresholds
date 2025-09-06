import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy.typing as npt

class conjugate_normal():
    def __init__(self, mu, k, a, b, percentile=.20, max_k=None, max_alpha=None, k_decay=None, alpha_decay=None, beta_decay=None):
        '''
        mu: prior mean
        k: uncertainty about the prior mean (pseudo-samples)
        alpha: shape of variance distribution
        beta: scale of variance distribution

        '''
        self.mu = mu
        self.k = k
        self.max_k = np.inf if not max_k else max_k
        self.alpha = a
        self.max_alpha = max_alpha
        self.k_decay = k_decay
        self.alpha_decay = alpha_decay
        self.beta_decay = beta_decay
        self.beta = b
        self.percentile = percentile

    def __repr__(self):
        return (
            f"ConjugateNormal("
            f"mu={self.mu:}, "
            f"k={self.k}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta:}, "
            f"percentile={self.percentile:.2f}, "
            f"sigma={self.sigma:.4f}, "
            f"threshold={self.threshold:.4f})"
        )

    def _apply_constraints(self):
        '''Apply constraints to prevent over-confidence'''
        if self.max_k:
            self.k = min(self.k, self.max_k)
        # Also bound alpha to prevent variance estimate from becoming too precise
        if self.max_alpha:
            self.alpha = min(self.alpha, self.max_alpha)

    def _apply_forgetting(self):
        '''Apply the forgetting strategy'''
        # Exponentially decay parameters
        if self.k_decay:
            self.k *= self.k_decay
        if self.alpha_decay:
            self.alpha = 1 + (self.alpha - 1) * self.alpha_decay
        if self.beta_decay:
            self.beta *= self.beta_decay

    def update(self, x: npt.ArrayLike):
        '''x: data
        '''
        self._apply_forgetting()

        self.mu = self.posterior_mean(x)
        self.alpha = self.posterior_alpha(x)
        self.beta = self.posterior_beta(x)
        self.k += len(x)

        # Apply post-update constraints
        self._apply_constraints()
        
    def posterior_mean(self, x):
        n = len(x)
        numerator = (self.k * self.mu) + (n * np.mean(x))
        denominator = self.k + n
        return numerator / denominator
    
    def posterior_alpha(self, x):
        n = len(x)
        return self.alpha + (n/2)
    
    def posterior_beta(self, x):
        n = len(x)
        mu_1 = np.mean(x)
        ssd = self.sum_square_diffs(x, mu_1)
        sd = (mu_1 - self.mu)**2
        return self.beta + 0.5*(ssd + (self.k*n*sd)/(self.k+n))

    @property
    def sigma(self):
        assert self.alpha > 1, "alpha must be greater than 1"
        return np.sqrt(self.beta / (self.alpha - 1))

    @property
    def dist(self):
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (1 + 1 / self.k) / self.alpha)
        return stats.t(df=df, loc=self.mu, scale=scale)

    @property
    def threshold(self):
        df = 2 * self.alpha
        scale = np.sqrt(self.beta * (1 + 1 / self.k) / self.alpha)
        return stats.t.ppf(self.percentile, df=df, loc=self.mu, scale=scale)

    def sum_square_diffs(self, A, B):
        '''Sum of squared differences'''
        return np.sum((A - B) ** 2)