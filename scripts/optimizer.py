import os
import sys
sys.path.append(r'/Users/benjaminshmulevsky/repos/Portfolio-Optimizer-DAO/') # Fix this later

import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
from utils import contract_utils as cu
from utils.portfolio_optimizer_utils import get_crypto_time_series, get_covariance_matrix_time_series

class MeanVarianceOptimizer:

    def __init__(self, tickers, expected_returns, cov_matrix, bounds):

        # Parameters for optimization
        self.tickers = tickers
        self.N = len(self.tickers)
        self.w = cp.Variable(self.N)
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix

        # # Risk limit
        # assert isinstance(risk_limit, (float, int)) and risk_limit > 0
        # self.vol_min = np.sqrt(1/np.sum(np.linalg.pinv(self.cov_matrix)))
        # self.risk_limit = self.vol_min if risk_limit < self.vol_min else risk_limit

        # Objective Function
        self.mu = self.w @ self.expected_returns
        self.objective = self.getObjectiveValue(self.w, self.mu)

        # Bounds
        if isinstance(bounds, (tuple, list)) \
                and isinstance(bounds[0], (int, float)) \
                and isinstance(bounds[1], (int, float)) \
                and len(bounds) == 2:
            self.lower_bound, self.upper_bound = bounds
        else:
            raise TypeError("Bounds must be a 2 element list.")

        # Constaints
        self.constraints = self.setConstraints()

        # Run optimization
        self.optimum = cp.Problem(cp.Minimize(self.objective), self.constraints)
        self.optimum.solve()
        assert self.optimum.status in ['optimal', 'optimal_inaccurate']

        # Compute weights
        self.weights = self.w.value.round(16) + 0.0
        self.optimal_weights = {t: w.round(4) for t, w in zip(self.tickers, self.weights)}

    def addConstraint(self, newConstraint):
        return newConstraint(self.w)

    def setConstraints(self):
        constraints = []

        # Bounds
        constraints.append(self.addConstraint(lambda x: x >= self.lower_bound))
        constraints.append(self.addConstraint(lambda x: x <= self.upper_bound))

        # Weights sum to 1
        constraints.append(self.addConstraint(lambda x: cp.sum(x) == 1))

        # Risk constraint?

        return constraints

    def getObjectiveValue(self, w, obj):
        if isinstance(w, np.ndarray):
            if np.isscalar(obj):
                return obj
            elif np.isscalar(obj.value):
                return obj.value
            else:
                return obj.value.item()
        else:
            return obj

class BlackLittermanOptimizer(MeanVarianceOptimizer):
    def __init__(self,
                 tickers,
                 views,
                 views_confidences=None,
                 tau=0.05,
                 risk_aversion=1.0,
                 risk_free=0.02,
                 prior=True,
                 decay_factor=0.97,
                 window=(1, 'years'),
                 shrinkage=True,
                 long_only=True):
        """
        :param tickers: list of N assets in the portfolio
        :type tickers: list of str
        :param views: a collection of K < N views (either absolute or relative)
        :type views: dictionary:
            keys <- str, name of tickers
            values <- tuple, either one or two elements:
                1) float, return for provided ticker (absolute return if second element not provided)
                2) str, ticker for relative return
        :param views_confidences (optional): confidence levels for each view. If None, Omega is calculated without confidences.
        :type views_confidences: float or dict:
            1) float, global confidence level for all views
            2) dict, confidence levels for each ticker. Keys must the views dictionary, values are float.
        :param tau (optional): controls weight on the views, preset to 0.05
        :type tau: float
        :param risk_aversion (optional): risk aversion parameter for computation of BL posterior weights, preset to 1
        :type risk_aversion: float
        :param risk_free (optional): risk-free return to include in prior return estimate, preset to 0.02
        :type risk_free: float
        :param prior (optional): inclusion of prior in the model, preset to True
        :type prior: bool
        :param decay_factor (optional): decay factor for EWM covariance estimation, preset to 0.97
        :type decay_factor: float
        :param window (optional): moving window for EWM covariance estimation, preset to 1 year
        :type window: tuple, first element is str (e.g. 'years', 'months', 'days'), second element is int
        :param shrinkage (optional): applies shrinkage to EWM covariance estimate via eigenvalue spectrum fit to the Marchenko-Pastur distribution
        :type shrinkage: bool
        """
        # Get data
        self.data = {}
        for ticker in tickers:
            self.data[ticker] = get_crypto_time_series(ticker)

        # Attributes
        self.tickers = tickers
        self.views = views
        self.views_confidences = views_confidences
        self.N = len(self.tickers)
        self.K = len(self.views)
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free = risk_free
        self.cov_matrix = self.getCovarianceMatrix(decay_factor, window, shrinkage)
        self.prior = prior
        self.long_only = long_only
        self.bounds = (0, 0.5) if self.long_only else (-1, 1)  # Do these make sense?

        # Check attribute errors
        self.checkErrors()

        # Calculated properties
        self.Q, self.P = self.parseViews(self.views)
        self.pi = self.priorReturns()
        self.omega = self.buildOmega(self.views_confidences)
        self.posterior_returns = self.posteriorReturns()
        self.posterior_covariance = self.posteriorCovariance()
        self.bl_optimal_weights = self.optimalBLWeights()
        
        # Run MVO on posterior returns
        super().__init__(self.tickers, self.posterior_returns, self.posterior_covariance, self.bounds)

########################################################################################################################
# Black-Litterman Model Parameters
########################################################################################################################

    def parseViews(self, views):
        """
        :param views: a collection of K < N views (either absolute or relative)
        :return: Kx1 vector of views ("Q"), KxN matrix of picks ("P")
        """
        # Raise error if not type dict or series
        if not isinstance(views, dict):
            raise TypeError("views should be a dict")

        if len(list(views.keys())) == self.N:
            # If a view on every asset is provided, set P to the identity matrix
            Q = np.array([i[0] for i in list(views.values())])
            P = np.eye(self.N)
        else:
            # Initialize variables
            views_series = pd.Series(views)  # Coerce to pandas series
            Q = np.zeros((self.K, 1))
            P = np.zeros((self.K, self.N))

            for i, view in enumerate(views_series.items()):
                ticker = view[0] if view[0] not in list(cu.contract_tokens_mapping.keys()) else cu.contract_tokens_mapping[view[0]]
                ticker_view = view[1]
                if len(ticker_view[1]) == 0:  # Case of absolute views
                    Q[i] = views_series[ticker][0]
                    P[i, list(self.tickers).index(ticker)] = 1
                else:  # Case of relative views
                    ticker_relative = ticker_view[1] if ticker_view[1] not in list(cu.contract_tokens_mapping.keys()) else cu.contract_tokens_mapping[ticker_view[1]]
                    Q[i] = views_series[ticker][0]
                    P[i, list(self.tickers).index(ticker)] = 1
                    P[i, list(self.tickers).index(ticker_relative)] = -1

        # Final sanity check
        assert Q.shape == (self.K, 1)
        assert P.shape == (self.K, self.N)

        return Q, P

    def priorReturns(self):
        """
        :return: Prior, Nx1 vector of market cap weighted returns estimated from the market ("pi")
        """
        if not self.prior:
            warnings.warn("Running Black-Litterman with no prior.")
            return np.zeros((self.N, 1))
        else:
            mkt_caps = self.getMarketCaps()

            # Set the prior returns to the implied returns from the market cap weighted portfolio and risk aversion
            mkt_weights = mkt_caps / mkt_caps.sum()
            pi = self.risk_aversion * self.cov_matrix.dot(mkt_weights) + self.risk_free
            pi = pi.reshape(-1, 1)

            # Final sanity check
            assert pi.shape == (self.N, 1)

            return pi

    def buildOmega(self, conf):
        """
        :param conf: confidence levels for each view
        :return: KxK diagonal uncertainty matrix calculated via Izdorek's Method ("Omega").
        """
        if conf is None:
            omega = np.diag(np.diag(self.tau * self.P @ self.cov_matrix @ self.P.T))
        else:
            view_omegas = []
            for view_idx, ticker in enumerate(list(self.views.keys())):
                view_conf = conf[ticker] if type(conf).__name__ == 'dict' else conf
                P_view = self.P[view_idx].reshape(1, -1)
                alpha = (1 - view_conf) / view_conf
                omega = self.tau * alpha * P_view @ self.cov_matrix @ P_view.T
                view_omegas.append(omega.item())
            omega = np.diag(view_omegas)

        # Final sanity check
        assert omega.shape == (self.K, self.K)

        return omega

    def getMarketCaps(self):
        """
        :return: array of market caps corresponding to the tickers property.
        """
        mkt_caps = []
        for ticker in self.tickers:
            ticker_data = self.data[ticker]
            ticker_mkt_cap = ticker_data.iloc[-1]['market cap']
            mkt_caps.append(ticker_mkt_cap)
        return np.array(mkt_caps)

    def getCovarianceMatrix(self, decay_factor, window, shrinkage):
        """
        :param decay_factor: decay factor for EWM covariance estimation, preset to 0.97
        :param window: moving window for EWM covariance estimation, preset to 1 year
        :param shrinkage: applies shrinkage to EWM covariance estimate via eigenvalue spectrum fit to the Marchenko-Pastur distribution
        :return: Most recent covariance matrix estimated from historical market returns
        """
        # for ticker in self.tickers:
        #     print(ticker)
        #     print(self.data[ticker])
        #     print(self.data[ticker]['price'])
        #     print('\n')
        price_data = pd.DataFrame({ticker: self.data[ticker]['price'] for ticker in self.tickers})
        returns_data = (price_data / price_data.shift(1)).dropna()
        covariance_matrix_ts = get_covariance_matrix_time_series(returns_data, decay_factor, window, shrinkage)
        covariance_matrix = covariance_matrix_ts.loc[covariance_matrix_ts.index == pd.Series(covariance_matrix_ts.index.to_list()).max()].to_numpy()

        # Final sanity check
        assert covariance_matrix.shape == (self.N, self.N)

        return covariance_matrix

########################################################################################################################
# Posterior Estimates
########################################################################################################################

    def posteriorReturns(self):
        """
        :return: Nx1 posterior estimate of returns vector
        """
        tau_sigma_P = self.tau * self.cov_matrix @ self.P.T
        A = (self.P @ tau_sigma_P) + self.omega
        b = self.Q - self.P @ self.pi
        post_rets = self.pi + tau_sigma_P @ np.linalg.solve(A, b)
        return pd.Series(post_rets.flatten(), index=self.tickers)
    
    def posteriorCovariance(self):
        """
        :return: NxN posterior estimate of covariance matrix
        """
        tau_sigma_P = self.tau * self.cov_matrix @ self.P.T
        A = (self.P @ tau_sigma_P) + self.omega
        b = tau_sigma_P.T
        M = self.tau * self.cov_matrix - tau_sigma_P @ np.linalg.solve(A, b)
        posterior_cov = self.cov_matrix + M
        return pd.DataFrame(posterior_cov, index=self.tickers, columns=self.tickers)

########################################################################################################################
# Weights
########################################################################################################################

    def optimalBLWeights(self):
        """
        :return: dictionary of optimal weights implied by the posterior returns corresponding to each asset.
                 (Gross sum may exceed 100%).e
        """
        A = self.risk_aversion * self.cov_matrix
        raw_weights = np.linalg.solve(A, self.posterior_returns)
        weights = raw_weights / raw_weights.sum()
        weights_dict = {ticker: weight for ticker, weight in zip(self.tickers, weights)}
        return weights_dict

    def normalizedWeights(self):
        """
        :return: dictionary of weights normalized to 100%, derived from the optimal weights corresponding to each asset.
                 E.g. an optimal weight of -150% for short BTC in a fund with a total gross weight of 300% is assumed
                 to be equivalent to a 50% long position in HEDGE.
                 The values are 2-element tuples:
                    1) float,  absolute weight as a percentage of the absolute gross sum of the optimal weights
                    2) string, long or short flag
        """
        gross_weight = 0
        for wt in list(self.optimalWeights.values()):
            gross_weight += abs(wt)
        normalized_weights = {ticker: (abs(weight)/gross_weight, self.ls(ticker))
                              for ticker, weight in self.optimalWeights.items()}
        return normalized_weights
    
    def postProcessWeights(self, optimalWeights):
        """
        :param optimalWeights: optimalWeights dictionary.
        :return: dictionary of long-only weights, derived from the long only weights corresponding to each asset.
        The values are 2-element tuples:
            1) float, weight
            2) string, long or short flag
        """

        long_positions = {t: w for t, w in list(optimalWeights.items()) if w >= 0}
        short_positions = {t: -w for t, w in list(optimalWeights.items()) if w < 0}
        if len(short_positions) == 0:
            return optimalWeights
        else:
            # Compute short ratio
            if self.short_ratio is None:
                short_ratio = sum([-w for w in list(optimalWeights.values()) if w < 0])
            else:
                short_ratio = self.short_ratio

        # Normalize weights
        total_long = sum(long_positions.values())
        total_short = sum(short_positions.values())
        long_positions = {t: w / total_long for t, w in list(long_positions.items())}
        short_positions = {t: w / total_short for t, w in list(short_positions.items())}
        long_value = self.portfolio_value
        short_value = self.portfolio_value * short_ratio

        # Run IP for longs/shorts separately?

        # Set up linear program
        full_allocation = {t: 0 for t in zip(list(optimalWeights.keys()))}
        for positions in [long_positions, short_positions]:
            p = np.array([get_crypto_time_series(t)['price'].iloc[-1] for t in list(optimalWeights.keys())])
            w = np.array(list(positions.values()))
            x = cp.Variable(len(p), integer=True)
            u = cp.Variable(len(p))
            total_value = long_value if positions == long_positions else short_value  # ToDo: fix this logic
            r = total_value - p.T @ x  # Remaining amount
            eta = w * self.portfolio_value - cp.multiply(x, p)
            constraints = [eta <= u, eta >= -u, x >= 0, r >= 0]
            objective = cp.sum(u) + r

            optimizer = cp.Problem(cp.Minimize(objective), constraints)
            optimizer.solve(solver="ECOS_BB")
            values = np.rint(x.value).astype(int)
            allocation = {t: w for t, w in zip(list(positions.keys()), values) if w != 0}
            full_allocation.update(allocation)

        return full_allocation

    def ls(self, ticker):
        """
        :param ticker: ticker name
        :return: long/short flag based on the sign of the optimal weight for the given ticker
        """
        return 'long' if self.optimalWeights[ticker] >= 0 else 'short'

########################################################################################################################
# Checks
########################################################################################################################

    def checkErrors(self):
        """
        Checks the data types and dimensions of various model inputs.
        """
        if self.cov_matrix.shape != (self.N, self.N):
            raise ValueError("Covariance matrix must be N X N")
        if self.risk_aversion < 0 or type(self.risk_aversion).__name__ != 'float':
            raise ValueError("Risk Aversion must be a float greater than zero")
        if self.tau <= 0 or self.tau > 1 or type(self.tau).__name__ != 'float':
            raise ValueError("Tau must be a float in (0, 1]")
        if type(self.views_confidences).__name__ not in ['float', 'dict']:
            raise ValueError("Confidence views must be a float or a dict.")
        if type(self.views_confidences).__name__ == 'float':
            if self.views_confidences <= 0 or self.views_confidences > 1:
                raise ValueError("View confidences must be in (0, 1]")
        if type(self.views_confidences).__name__ == 'dict':
            for view_confidence in list(self.views_confidences.values()):
                if view_confidence <= 0 or view_confidence > 1:
                    raise ValueError("View confidences must be in (0, 1]")

# # Uncomment to create a sample BlackLittermanOptimizer object
if __name__ == '__main__':
    tickers = ['BTC', 'MKR', 'UNI', 'BAT']
    views = {'BTC': (0.02, 'MKR'), 'UNI': (0.03, '')}
    views_confidences = {'BTC': 1.0, 'UNI': 1.0}
    x = BlackLittermanOptimizer(tickers, views, views_confidences)
    print('Black-Litterman Optimal Weights:', x.bl_optimal_weights)
    print('Constrained MVO weights:', x.optimal_weights)