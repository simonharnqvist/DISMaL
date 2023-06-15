from dismal.inference import DemographicModel


def fit_all(S):

    models = {}

    models["iso_2stage"] = DemographicModel(
        model_name="iso_2stage", pop_size_change=False, migration=False).infer(S=S)
    models["im_2stage"] = DemographicModel(
        model_name="im_2stage", pop_size_change=False, migration=True).infer(S=S)
    models["im_2stage_asym"] = DemographicModel(
        model_name="im_2stage_asym", pop_size_change=False, migration=True, asymmetric_m=True).infer(S=S)
    models["iso_3stage"] = DemographicModel(
        model_name="iso_3stage", pop_size_change=True, migration=False).infer(S=S)
    models["iim_sym"] = DemographicModel(model_name="iim_sym", pop_size_change=True, migration=True, mig_rate_change=True, remove_mig_params=[
        "M_prime_star"]).infer(S=S)
    models["sc_sym"] = DemographicModel(
        model_name="sc_sym", pop_size_change=True, migration=True, mig_rate_change=True, remove_mig_params=["M_star"]).infer(S=S)
    models["gim_1m"] = DemographicModel(
        model_name="gim_1m", pop_size_change=True, migration=True).infer(S=S)
    models["iim_asym"] = DemographicModel(model_name="iim_asym", pop_size_change=True, migration=True, mig_rate_change=True,
                                          asymmetric_m_star=True, remove_mig_params=["M_prime_star"]).infer(S=S)
    models["sc_asym"] = DemographicModel(model_name="sc_asym", pop_size_change=True, migration=True, mig_rate_change=True,
                                         asymmetric_m_prime_star=True, remove_mig_params=["M_star"]).infer(S=S)
    models["gim_sym"] = DemographicModel(
        model_name="gim_sym", pop_size_change=True, migration=True, mig_rate_change=True).infer(S=S)
    models["gim_asym1"] = DemographicModel(
        model_name="gim_asym1", pop_size_change=True, migration=True, mig_rate_change=True, asymmetric_m_star=True).infer(S=S)
    models["gim_asym2"] = DemographicModel(
        model_name="gim_asym2", pop_size_change=True, migration=True, mig_rate_change=True, asymmetric_m_prime_star=True).infer(S=S)
    models["gim_asym"] = DemographicModel(model_name="gim_asym", pop_size_change=True, migration=True,
                                          mig_rate_change=True, asymmetric_m_star=True, asymmetric_m_prime_star=True).infer(S=S)

    return models
