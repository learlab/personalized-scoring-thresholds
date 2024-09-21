import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy.typing as npt

class conjugate_normal():
    def __init__(self, mu, k, a, b, percentile=.40):
        '''
        mu: prior mean
        k: uncertainty about the prior mean (pseudo-samples)
        alpha: shape of variance distribution
        beta: scale of variance distribution

        '''
        self.mu = mu
        self.k = k
        self.alpha = a
        self.beta = b
        self.xlim = np.linspace(self.mu - 3*self.sigma, self.mu + 3*self.sigma, 1001) # For plotting
        self.percentile = percentile

    def update(self, x: npt.ArrayLike):
        '''x: data
        '''
        self.mu = self.posterior_mean(x)
        self.alpha = self.posterior_alpha(x)
        self.beta = self.posterior_beta(x)
        self.k += len(x)
        
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
    def threshold(self):
        return stats.norm.ppf(self.percentile, loc=self.mu, scale=self.sigma)

    def sum_square_diffs(self, A, B):
        '''Sum of squared differences'''
        squared_differences = (A - B) ** 2
        return np.sum(squared_differences)

    def plot(self, u=3.0, draw_percentile=True, color="blue", **kwargs):
        dist = stats.norm(self.mu, self.sigma)
        plt.plot(self.xlim, dist.pdf(self.xlim), color=color, **kwargs)

        # Add percentile indicator
        if draw_percentile:
            plt.axvline(
                ppf,
                color=color,
                linestyle="--",
                label=f'{int(self.percentile*100):2d}th Percentile: {self.threshold:.2f}'
            )