def aic(lnl, n_params):
    aic = 2*(n_params-lnl)
    return aic