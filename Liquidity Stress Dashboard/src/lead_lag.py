import numpy as np

def lead_lag_correlation(early_index, late_index, max_lag =20):
    corrs = []
    for lag in range(0, max_lag +1):
        corr = early_index[:-lag].corr(late_index[lag:]) if lag != 0 else early_index.corr(late_index)
        corrs.append(corr)
    max_corr = max(corrs)
    best_lag  = corrs.index(max_corr)
    return best_lag, max_corr, corrs
