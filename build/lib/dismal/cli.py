#!/usr/bin/env python
import json
from argparse import ArgumentParser
from dismal.s_matrix import SMatrix
from dismal.model_fitting import fit_all
from dismal.metrics import best_fit_model


def arguments_input():
    parser = ArgumentParser()
    parser.add_argument(
        "--s1", help="Counts of segregating sites per block in population 1", required=True)
    parser.add_argument(
        "--s2", help="Counts of segregating sites per block in population 2", required=True)
    parser.add_argument(
        "--s3", help="Counts of segregating sites per block between populations 1 and 2", required=True)
    # parser.add_argument("--vcf")
    # parser.add_argument("--zarr_path", help="Path to Zarr store", default=".")
    parser.add_argument("--model_selection_method", default="aic")
    parser.add_argument("--results_json", required=False)
    args = parser.parse_args()

    return args


def get_input(s1, s2, s3):
    S = SMatrix().from_rgim_simulation([s1, s2, s3])
    return S


def main():
    args = arguments_input()
    S = get_input(args.s1, args.s2, args.s3)
    models = fit_all(S)
    best_mod, best_mod_aic = best_fit_model(models)
    inferred_parameters = models[best_mod][0]
    res = {"inferred_model_name": best_mod,
           "AIC": best_mod_aic,
           "inferred_parameters": inferred_parameters}

    print(res)

    if args.results_json is not None:
        with open(args.results_json, 'r+') as f:
            dic = json.load(f)
            dic.update(res)
            json.dump(dic, f)


if __name__ == "__main__":
    main()
