#!/usr/bin/env python3
"""Train the models once and save everything the dashboard needs to model.pkl.
Run:  python build_model.py --postings data/postings.csv --out model.pkl
(needs skillgap_core.py and career_skill_gap_analysis.py in the parent folder or the same folder)
"""
import argparse, os, sys, json, joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    sys.path.insert(0, p)
import skillgap_core as K
import career_skill_gap_analysis as A

ap = argparse.ArgumentParser()
ap.add_argument("--postings", default="data/postings.csv")
ap.add_argument("--cache", default="outputs/cache")
ap.add_argument("--salary_csv", default="outputs/salary_coefficients_python.csv")
ap.add_argument("--results", default="outputs/results.json")
ap.add_argument("--out", default="model.pkl")
ap.add_argument("--cap", type=int, default=800)
a = ap.parse_args()

post, strength, exact = K.load_postings(a.postings, a.cache)
y_all = post["role"].values
rng = np.random.default_rng(A.SEED)
keep = np.sort(np.concatenate([rng.permutation(np.where(y_all == k)[0])[:a.cap] for k in range(K.R_N)]))
Xb = (strength[keep] > 0).astype(float)
y = y_all[keep]

imp_all, _ = A.importance(strength, y_all)          # market database for scoring (all IT postings)
imp_train, _ = A.importance(strength[keep], y)      # weights the classifier's coverage term uses
alpha = 0.7
if os.path.exists(a.results):
    alpha = float(json.load(open(a.results))["alpha"])
M = A.Models(imp_train).fit(Xb, y)
pop = (strength > 0).mean(0)

sal = pd.read_csv(a.salary_csv) if os.path.exists(a.salary_csv) else pd.DataFrame(columns=["skill", "pct", "t", "n"])
bundle = dict(imp_all=imp_all, M=M, alpha=alpha, pop=pop, salary=sal[["skill", "pct", "t", "n"]].to_dict("records"),
              n_postings=int(len(post)), n_per_role={K.ROLE_NAMES[k]: int((y_all == k).sum()) for k in range(K.R_N)})
joblib.dump(bundle, a.out, compress=("xz", 6))
print("saved", a.out, "| alpha", alpha, "| postings", len(post))
