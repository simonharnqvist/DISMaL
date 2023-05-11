from scipy.stats import chi2

def aic(lnl, n_params):
    aic = 2*(n_params-lnl)
    return aic

def likelihood_ratio_test(model1, model2, alpha):

    lnl_h0 = -model1.negll
    lnl_h1 = -model2.negll
    n_params = model2.n_params - model1.n_params

    lr = -2*(lnl_h0-lnl_h1)
    crit_val = chi2.ppf(1-alpha, n_params)

    if lr > crit_val:
        reject_h0 = True
    else:
        reject_h0 = False

    return {"LR": lr, "Critical value": crit_val, "Reject H0": reject_h0}