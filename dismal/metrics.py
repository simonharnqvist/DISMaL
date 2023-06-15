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


def best_fit_model(models, true_mod=None):
    """needs AIC threshold"""
    mod_names = list(models.keys())
    likelihoods = [models[mod][1] for mod in models]
    n_params = [models[mod][2] for mod in models]
    params_lnl = [i for i in zip(likelihoods, n_params)]
    models_dict = dict(
        zip(mod_names, [aic(-lnl, n_params) for (lnl, n_params) in params_lnl]))

    if true_mod:
        true_aic = models_dict[true_mod]
    else:
        true_aic = None

    best_fit_mod = min(models_dict, key=models_dict.get)
    best_fit_aic = models_dict[best_fit_mod]

    return best_fit_mod, best_fit_aic
