import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import requests

# API Utils
def get_crypto_time_series(symbol, exchange='USD', start_date=None):
    """
    :param symbol: str, coin ticker
    :param exchange: str, default to USD
    :param start_date: str, date
    :return: dataframe of price and market cap from AlphaVantage API
    """
    api_key = '2HT2PCFAZ56G0383'
    api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1][['4a. close (USD)', '6. market cap (USD)']]
    df.rename(columns={'4a. close (USD)': 'price', '6. market cap (USD)': 'market cap'}, inplace=True)
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    return df

# Statistics
def get_covariance_matrix_time_series(df, decay_factor, window, shrinkage):
    """
    :param df: (m,n) dataframe of returns
    :param decay_factor: float, decay factor for exponentially weighted moving covariance
    :param window: tuple, length of time series for the covariance matrix estimation. Example: (1, 'years')
    :param shrinkage: bool, if True, shrinks daily covariance matrix via Marchenko-Pastur fit
    :return: (m*n, n) dataframe of residuals, indexed by date
    """
    print("Estimating covariance matrix time series...")
    if window[1] == 'years':
        window_len = df.loc[df.index < (df.index.min() + pd.DateOffset(years=window[0]))].shape[0]
    elif window[1] == 'months':
        window_len = df.loc[df.index < (df.index.min() + pd.DateOffset(months=window[0]))].shape[0]
    elif window[1] == 'days':
        window_len = window
    else:
        raise ValueError("Improper window specification.")
    cov_matrix_ts = (df.ewm(alpha=decay_factor, min_periods=window_len).cov().reset_index()
                     .rename(columns={'level_0': 'Date'}).dropna())
    cov_matrix_ts = cov_matrix_ts.set_index('Date')
    cov_matrix_ts = cov_matrix_ts.drop('level_1', axis=1)

    if shrinkage:
        print("Shrinking covariance matrices via Marchenko-Pastur fit...")
        dates = cov_matrix_ts.index.unique().tolist()
        daily_cov_matrix_l = []
        for dt in dates:
            daily_cov_matrix = cov_matrix_ts.loc[dt]
            daily_cov_matrix_s = spectrum_shrink(daily_cov_matrix.to_numpy(), 12)
            df_daily_cov_matrix_s = pd.DataFrame(daily_cov_matrix_s,
                                                 columns=daily_cov_matrix.columns.to_list(),
                                                 index=daily_cov_matrix.index.to_list())
            daily_cov_matrix_l.append(df_daily_cov_matrix_s)
        cov_matrix_ts = pd.concat(daily_cov_matrix_l)

    return cov_matrix_ts

def spectrum_shrink(sigma2_base, t):
    """
    :param sigma2_base: nxn base covariance matrix estimate
    :param t: length of time series
    :return: nxn covariance matrix estimate with shrinkage applied by fitting eigenvalue spectrum to the Marchenko-Pastur distribution
    """

    n = sigma2_base.shape[0]

    # PCA decomposition
    e, lambda2 = pca_cov(sigma2_base)

    # Determine optimal k
    ll = 1000
    dist = np.ones(n - 1) * np.nan
    for k in range(n - 1):
        lambda2_k = lambda2[k + 1:]
        lambda2_noise = np.mean(lambda2_k)
        q = t / len(lambda2_k)
        x_tmp, mp_tmp, x_lim = marchenko_pastur(q, ll, lambda2_noise)
        if q > 1:
            x_tmp = np.r_[0, x_lim[0], x_tmp]
            mp_tmp = np.r_[0, mp_tmp[0], mp_tmp]
        l_max = np.max(lambda2_k)
        if l_max > x_tmp[-1]:
            x_tmp = np.r_[x_tmp, x_lim[1], l_max]
            mp_tmp = np.r_[mp_tmp, 0, 0]

        # compute the histogram of eigenvalues
        hgram, x_bin_edge = np.histogram(lambda2_k, bins='auto', density=True)
        bin_size = np.diff(x_bin_edge)[0]
        x_bin = x_bin_edge[:-1] + bin_size / 2

        # interpolation
        interp = interp1d(x_tmp, mp_tmp, fill_value='extrapolate')
        mp = interp(x_bin)

        dist[k] = np.mean((mp - hgram) ** 2)

    err_tmp, k_tmp = np.nanmin(dist), np.nanargmin(dist)
    k_opt = k_tmp

    # Isotropy
    lambda2_out = lambda2
    lambda2_noise = np.mean(lambda2[k_opt + 1:])

    # shrunk spectrum
    lambda2_out[k_opt + 1:] = lambda2_noise

    # Output
    sigma2_out = e @ np.diagflat(lambda2_out) @ e.T

    return sigma2_out

def pca_cov(sigma2):
    """
    :param sigma2: nxn covariance matrix
    :return: PCA decomposition of covariance matrix into k eigenvalues sorted descending and corresponding eigenvectors
    """
    
    lambda2, e = np.linalg.eigh(sigma2)
    lambda2 = lambda2[::-1]
    e = e[:, ::-1]
    ind = np.argmax(abs(e), axis=0)
    ind = np.diag(e[ind, :]) < 0
    e[:, ind] = -e[:, ind]

    return e, lambda2

def marchenko_pastur(q, ll, sigma2=1):
    """
    Computes the Marchenko Pastur distribution.
    """
    eps = 1e-9

    # Ensure sigma2 is not too small
    if sigma2 < 10 * eps:
        sigma2 = 10 * eps

    xlim = np.array([(1 - 1 / np.sqrt(q)) ** 2, (1 + 1 / np.sqrt(q)) ** 2]) * sigma2
    xlim_tmp = [0, 0]
    if q > 1:
        xlim_tmp[1] = xlim[1] - eps
        xlim_tmp[0] = xlim[0] + eps
        dx = (xlim_tmp[1] - xlim_tmp[0]) / (ll - 1)
        x = xlim_tmp[0] + dx * np.arange(ll)
        y = q * np.sqrt((4 * x) / (sigma2 * q) - (x / sigma2 + 1 / q - 1) ** 2) / (2 * np.pi * x)
    elif q < 1:
        xlim_tmp[1] = xlim[1] - eps
        xlim_tmp[0] = xlim[0] + eps
        dx = (xlim_tmp[1] - xlim_tmp[0]) / (ll - 2)
        x = xlim_tmp[0] + dx * np.arange(ll - 1)
        y = q * np.sqrt((4 * x) / (sigma2 * q) - (x / sigma2 + 1 / q - 1) ** 2) / (2 * np.pi * x)
        xlim[0] = 0
        x = [0, x]
        y = [(1 - q), y]
    else:
        xlim = np.array([0, 4]) * sigma2
        dx = xlim[1] / ll
        x = dx * np.arange(1, ll)
        y = np.sqrt(4 * x / sigma2 - (x / sigma2) ** 2) / (2 * np.pi * x)

    return x, y, xlim