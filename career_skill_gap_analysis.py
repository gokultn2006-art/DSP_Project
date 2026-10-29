#!/usr/bin/env python3
"""
Data-Driven Career Skill Gap Analyzer and Job Recommendation System  (CSI3004)
Real-data experiments.
"""
import argparse, json, os, re, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import skillgap_core as K
from skillgap_core import SKILLS, SIDX, S_N, ROLE_NAMES, R_N, CATEGORY

warnings.filterwarnings("ignore")
SEED = 42
plt.rcParams.update({"font.family": "serif", "font.size": 6.5, "axes.linewidth": 0.5, "axes.titlesize": 7, "legend.fontsize": 5.8})
C = ["#1f4e79", "#c0504d", "#4f8f4f", "#d98c1f", "#7a5a9e", "#555555"]
SHORT = ["DS", "ML/AI", "DA", "DE", "BA", "FullSt", "Front", "Mobile", "Sec", "DevOps", "QA", "DBA", "NetAd", "SWE"]

HOURS = dict.fromkeys(["Machine Learning"], 120) | {
    "Deep Learning": 100, "Statistics": 80, "System Design": 90, "Kubernetes": 60, "Spark": 60, "Java": 100, "Python": 60, "C++": 100, "C#": 80,
    "Scala": 80, "Rust": 100, "Go": 60, "TypeScript": 30, "JavaScript": 60, "SQL": 40, "R": 50, "Cryptography": 70, "Penetration Testing": 80,
    "SIEM": 50, "Network Security": 70, "Networking": 70, "AWS": 60, "Azure": 60, "GCP": 60, "Cloud Computing": 40, "Terraform": 40, "React": 50,
    "Angular": 60, "Vue.js": 45, "Android": 70, "iOS": 70, "Flutter": 50, "React Native": 50, "LLMs": 60, "Generative AI": 50, "NLP": 70,
    "Computer Vision": 70, "PyTorch": 50, "TensorFlow": 50, "Keras": 30, "Git": 12, "Excel": 20, "HTML": 20, "CSS": 25, "Agile": 10, "Docker": 25,
    "Jenkins": 25, "CI/CD": 30, "Linux": 40, "Bash": 20, "Project Management": 30, "Business Analysis": 30, "Stakeholder Management": 15,
    "Technical Support": 20, "Compliance": 25, "Data Structures": 80, "SDLC": 10, "Data Analysis": 40, "Data Visualization": 30, "Tableau": 25,
    "Power BI": 25, "Hadoop": 50, "Kafka": 40, "Airflow": 30, "ETL": 40, "Data Warehousing": 50, "Snowflake": 30, "Databricks": 35, "Selenium": 30,
    "Test Automation": 40, "Manual Testing": 20, "JUnit": 20, "Postman": 10, "JIRA": 8, "Performance Testing": 25, "Database Administration": 60,
    "MySQL": 25, "PostgreSQL": 30, "MongoDB": 30, "Oracle": 40, "SQL Server": 35, "Redis": 15, "NoSQL": 25, "Elasticsearch": 30, "DynamoDB": 20, "Cassandra": 30}
DEFAULT_HOURS = 35
PREREQ = {"Pandas": ["Python"], "NumPy": ["Python"], "Matplotlib": ["Python"], "Scikit-learn": ["Python"], "Machine Learning": ["Statistics"],
          "Deep Learning": ["Machine Learning"], "PyTorch": ["Deep Learning", "Python"], "TensorFlow": ["Deep Learning", "Python"], "Keras": ["Deep Learning", "Python"],
          "NLP": ["Machine Learning"], "Computer Vision": ["Deep Learning"], "LLMs": ["Deep Learning", "NLP"], "Generative AI": ["Deep Learning"],
          "MLOps": ["Docker", "Machine Learning"], "React": ["JavaScript", "HTML"], "TypeScript": ["JavaScript"], "Node.js": ["JavaScript"], "Angular": ["TypeScript"],
          "Vue.js": ["JavaScript"], "React Native": ["React"], "Spring Boot": ["Java"], "Django": ["Python"], "Flask": ["Python"], "Kubernetes": ["Docker"],
          "Spark": ["Python", "SQL"], "Airflow": ["Python"], "Kotlin": ["Java"], "Android": ["Java"], "iOS": ["Swift"], "Selenium": ["Test Automation"],
          "Terraform": ["Linux"], "Penetration Testing": ["Network Security"], "SIEM": ["Network Security"], "Tableau": ["Data Visualization"],
          "Power BI": ["Data Visualization"], "Data Warehousing": ["SQL"], "ETL": ["SQL"], "Snowflake": ["SQL"], "CI/CD": ["Git"], "Jenkins": ["Git"]}

# role families: 0 data, 1 software, 2 infrastructure / security
FAMILY = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 10: 1, 13: 1, 8: 2, 9: 2, 11: 2, 12: 2}


def hours_of(s): return HOURS.get(s, DEFAULT_HOURS)
def fam_acc(P, y): return float(np.mean([FAMILY[int(a)] == FAMILY[int(b)] for a, b in zip(P.argmax(1), y)]))


# Importance, readiness, planning
def importance(strength, y, thr=0.10, dfthr=0.15, idf_on=True):
    """I_r(s) = f_r(s) * idf(s); strength holds 1.0 (required) / 0.5 (preferred-only) mentions."""
    f = np.vstack([strength[y == k].mean(0) for k in range(R_N)])
    df = (f >= dfthr).sum(0)
    idf = np.log((1 + R_N) / (1 + df)) + 1 if idf_on else np.ones(S_N)
    imp = f * idf
    imp[f < thr] = 0.0
    return imp, f


def readiness(x, imp): return (x @ imp.T) / imp.sum(1)[None, :]


def expand(skill, held):
    out = []
    def rec(t):
        for p in PREREQ.get(t, []):
            if p not in held and p not in out: rec(p)
        if t not in held and t not in out: out.append(t)
    rec(skill)
    return out


def plan(held, role_i, imp, method, rng, pop):
    held = set(held)
    req = [SKILLS[j] for j in np.where(imp[role_i] > 0)[0] if SKILLS[j] not in held]
    if method == "random": rng.shuffle(req); order = req
    elif method == "importance": order = sorted(req, key=lambda s: -imp[role_i, SIDX[s]])
    elif method == "popularity": order = sorted(req, key=lambda s: -pop[SIDX[s]])
    else:
        order, cur, pool = [], set(held), list(req)
        while pool:
            def ratio(s):
                ex = expand(s, cur)
                if not ex: return -1.0
                return sum(imp[role_i, SIDX[t]] for t in ex) / sum(hours_of(t) for t in ex)
            b = max(pool, key=ratio); pool.remove(b)
            if b in cur: continue
            for t in expand(b, cur): cur.add(t)
            order.append(b)
    seq, cur = [], set(held)
    for s in order:
        for t in expand(s, cur):
            seq.append(t); cur.add(t)
    return seq


def gain_at(held_vec, seq, role_i, imp, budget):
    learned, used = set(), 0
    for t in seq:
        h = hours_of(t)
        if used + h > budget: break
        used += h; learned.add(t)
    x1 = held_vec.copy()
    for t in learned: x1[SIDX[t]] = 1
    return float((x1 - held_vec) @ imp[role_i] / imp[role_i].sum())


# Models
def softmax(z, k):
    z = k * z; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def topk(P, y, k): return float(np.mean([y[i] in np.argsort(-P[i])[:k] for i in range(len(y))]))
def mrr(P, y): return float(np.mean([1.0 / (1 + list(np.argsort(-P[i])).index(y[i])) for i in range(len(y))]))


def metrics(P, y, labels=None):
    pred = P.argmax(1)
    return dict(acc=accuracy_score(y, pred), f1=f1_score(y, pred, average="macro", labels=labels, zero_division=0), top3=topk(P, y, 3), mrr=mrr(P, y))


class Models:
    def __init__(self, imp, kappa=8.0, n_trees=200):
        self.imp, self.kappa = imp, kappa
        self.rf = RandomForestClassifier(n_trees, random_state=SEED, n_jobs=-1, min_samples_leaf=1)
        self.svm = SVC(kernel="rbf", C=5, gamma="scale", probability=True, random_state=SEED)
        self.lr = LogisticRegression(max_iter=3000, C=1.0)
        self.nb = BernoulliNB()

    def fit(self, X, y):
        for m in (self.rf, self.svm, self.lr, self.nb): m.fit(X, y)
        return self

    def cov(self, X): return softmax(readiness(X, self.imp), self.kappa)
    def cosine(self, X): return cosine_similarity(X, self.imp)
    def ml(self, X): return (self.rf.predict_proba(X) + self.svm.predict_proba(X)) / 2
    def proba(self, X, alpha): return alpha * self.ml(X) + (1 - alpha) * self.cov(X)


def savefig(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--postings", default="data/postings.csv")
    ap.add_argument("--resumes", default="data/resume_data.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--cap", type=int, default=800, help="max postings per role used for model training/testing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res, t0 = {}, time.time()

    #data
    post, st_all, ex_all = K.load_postings(a.postings, os.path.join(a.out, "cache"))
    raw_n = int(pd.read_csv(a.postings, usecols=["job_id"]).shape[0])
    rraw, ures = K.load_resumes(a.resumes)
    y_all = post["role"].values
    res["data"] = dict(raw_postings=raw_n, role_postings=len(post), roles=R_N, skills=S_N,
                       postings_per_role={ROLE_NAMES[k]: int((y_all == k).sum()) for k in range(R_N)},
                       resume_rows=len(rraw), unique_resumes=len(ures), labelled_resumes=int((ures["role"] >= 0).sum()),
                       posting_skills_exact=float((ex_all > 0).sum(1).mean()), posting_skills_alias=float((st_all > 0).sum(1).mean()),
                       posting_zero_exact=float(((ex_all > 0).sum(1) == 0).mean()), posting_zero_alias=float(((st_all > 0).sum(1) == 0).mean()),
                       preferred_share=float(((st_all == 0.5).sum() / max((st_all > 0).sum(), 1))))
    #balanced subset
    keep = np.concatenate([rng.permutation(np.where(y_all == k)[0])[:a.cap] for k in range(R_N)])
    keep = np.sort(keep)
    P_ = post.iloc[keep].reset_index(drop=True); S_ = st_all[keep]; E_ = ex_all[keep]; y = y_all[keep]
    Xb, Xe = (S_ > 0).astype(float), (E_ > 0).astype(float)
    res["data"]["balanced_postings"] = len(keep)
    tr, te = train_test_split(np.arange(len(keep)), test_size=0.30, stratify=y, random_state=SEED)
    imp, freq = importance(S_[tr], y[tr])
    res["data"]["requirement_set_sizes"] = {ROLE_NAMES[k]: int((imp[k] > 0).sum()) for k in range(R_N)}
    res["top_skills"] = {ROLE_NAMES[k]: [(SKILLS[j], round(float(imp[k, j]), 2)) for j in np.argsort(-imp[k])[:6]] for k in range(R_N)}

    #resumes: extraction variants
    def resume_vec(row, variant):
        s = set()
        for it in row["skill_items"]:
            m = K.map_item(it, variant)
            if m: s.add(m)
        if variant != "exact": s |= K.extract_text(row["narr"], "alias")
        return K.vec(s)
    RV = {v: np.vstack([resume_vec(r_, v) for _, r_ in ures.iterrows()]) for v in ("exact", "alias", "fuzzy")}
    items = [it for l in ures["skill_items"] for it in l]
    mapped = {v: float(np.mean([K.map_item(it, v) is not None for it in items])) for v in ("exact", "alias", "fuzzy")}
    lab0 = np.where(ures["role"].values >= 0)[0]
    items_t = [it for i_ in lab0 for it in ures["skill_items"].iloc[i_]]
    mapped_t = {v: float(np.mean([K.map_item(it, v) is not None for it in items_t])) for v in ("exact", "alias", "fuzzy")}
    res["resume_extraction"] = dict(items=len(items), mapped=mapped, items_tech=len(items_t), mapped_tech=mapped_t,
                                    skills_per_resume={v: float(RV[v].sum(1).mean()) for v in RV},
                                    zero_skill_share={v: float((RV[v].sum(1) == 0).mean()) for v in RV})
    Xr = RV["fuzzy"]                                    # proposed extractor for candidates

    #role classification on postings
    alphas = np.round(np.arange(0, 1.01, 0.1), 2)
    cvs = np.zeros(len(alphas))
    for a_, b_ in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xb[tr], y[tr]):
        imp_f, _ = importance(S_[tr][a_], y[tr][a_])
        mv = Models(imp_f, n_trees=100).fit(Xb[tr][a_], y[tr][a_])
        ml = mv.ml(Xb[tr][b_]); cv_ = mv.cov(Xb[tr][b_])
        for k, al in enumerate(alphas):
            cvs[k] += accuracy_score(y[tr][b_], (al * ml + (1 - al) * cv_).argmax(1)) / 3
    alpha = float(alphas[np.where(cvs >= cvs.max() - 1e-9)[0]].mean())
    res["alpha"] = alpha; res["alpha_cv"] = {str(k): float(v) for k, v in zip(alphas, cvs)}

    t = time.time(); M = Models(imp).fit(Xb[tr], y[tr]); res["train_seconds"] = time.time() - t
    t = time.time(); Pprop = M.proba(Xb[te], alpha); res["infer_ms"] = (time.time() - t) / len(te) * 1000

    tf = TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True).fit(P_["text"].values[tr])
    Ttr, Tte = tf.transform(P_["text"].values[tr]), tf.transform(P_["text"].values[te])
    cent = np.vstack([np.asarray(Ttr[y[tr] == k].mean(0)).ravel() for k in range(R_N)])
    lr_txt = LogisticRegression(max_iter=3000, C=10).fit(Ttr, y[tr])
    table = {"TF-IDF centroid match": cosine_similarity(Tte, cent), "TF-IDF + Logistic Reg. (full text)": lr_txt.predict_proba(Tte),
             "Cosine to role profile": M.cosine(Xb[te]), "Weighted coverage only": M.cov(Xb[te]), "Naive Bayes": M.nb.predict_proba(Xb[te]),
             "Logistic Regression": M.lr.predict_proba(Xb[te]), "Random Forest": M.rf.predict_proba(Xb[te]), "SVM (RBF)": M.svm.predict_proba(Xb[te]),
             "Proposed hybrid": Pprop}
    res["models"] = {k: metrics(v, y[te]) for k, v in table.items()}
    for k, v in table.items(): res["models"][k]["family"] = fam_acc(v, y[te])
    res["per_role_f1"] = {ROLE_NAMES[k]: dict(hybrid=float(v1), rf=float(v2)) for k, (v1, v2) in enumerate(zip(
        f1_score(y[te], Pprop.argmax(1), average=None, labels=range(R_N)), f1_score(y[te], table["Random Forest"].argmax(1), average=None, labels=range(R_N))))}
    # 5-fold CV (imp recomputed inside each fold)
    cvr = {k: [] for k in ("Random Forest", "SVM (RBF)", "Proposed hybrid")}
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xb, y):
        imp_f, _ = importance(S_[a_], y[a_])
        mm = Models(imp_f, n_trees=100).fit(Xb[a_], y[a_])
        cvr["Random Forest"].append(accuracy_score(y[b_], mm.rf.predict(Xb[b_])))
        cvr["SVM (RBF)"].append(accuracy_score(y[b_], mm.svm.predict(Xb[b_])))
        cvr["Proposed hybrid"].append(accuracy_score(y[b_], mm.proba(Xb[b_], alpha).argmax(1)))
    res["cv"] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in cvr.items()}
    res["alpha_sweep"] = {str(al): float(accuracy_score(y[te], M.proba(Xb[te], al).argmax(1))) for al in alphas}
    # extraction variant -> downstream accuracy (exact-name vs alias dictionary)
    ex_acc = {}
    for nm, XX in (("exact", Xe), ("alias", Xb)):
        imp_v, _ = importance((XX[tr] > 0).astype(float), y[tr])
        mm = Models(imp_v, n_trees=100).fit(XX[tr], y[tr])
        ex_acc[nm] = float(accuracy_score(y[te], mm.proba(XX[te], alpha).argmax(1)))
    res["extraction_downstream_acc"] = ex_acc
    # robustness to missing skills
    drops = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rob = {k: [] for k in ("Random Forest", "SVM (RBF)", "Weighted coverage only", "Proposed hybrid")}
    r2 = np.random.default_rng(7)
    for d in drops:
        Xn = Xb[te] * (r2.random(Xb[te].shape) >= d)
        rob["Random Forest"].append(accuracy_score(y[te], M.rf.predict(Xn)))
        rob["SVM (RBF)"].append(accuracy_score(y[te], M.svm.predict(Xn)))
        rob["Weighted coverage only"].append(accuracy_score(y[te], M.cov(Xn).argmax(1)))
        rob["Proposed hybrid"].append(accuracy_score(y[te], M.proba(Xn, alpha).argmax(1)))
    res["robustness"] = dict(drops=drops, **rob)
    # learning curve
    lc = []
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        sub = tr[: int(len(tr) * frac)]
        imp_s, _ = importance(S_[sub], y[sub])
        mm = Models(imp_s, n_trees=100).fit(Xb[sub], y[sub])
        lc.append((len(sub), accuracy_score(y[te], mm.rf.predict(Xb[te])), accuracy_score(y[te], mm.proba(Xb[te], alpha).argmax(1))))
    res["learning_curve"] = lc
    # temporal generalisation: train on earlier postings, test on later ones
    order = np.argsort(P_["listed_time"].values); half = len(order) // 2
    early, late = order[:half], order[half:]
    imp_t, _ = importance(S_[early], y[early])
    mt = Models(imp_t, n_trees=100).fit(Xb[early], y[early])
    res["temporal"] = dict(train_n=len(early), test_n=len(late), hybrid=float(accuracy_score(y[late], mt.proba(Xb[late], alpha).argmax(1))),
                           rf=float(accuracy_score(y[late], mt.rf.predict(Xb[late]))),
                           iid_hybrid=float(res["models"]["Proposed hybrid"]["acc"]),
                           span_days=float((P_["listed_time"].max() - P_["listed_time"].min()) / 86400000))

    #transfer to real resumes
    lab = np.where(ures["role"].values >= 0)[0]
    yr = ures["role"].values[lab]
    tr_res = {}
    for v in ("exact", "alias", "fuzzy"):
        tr_res[v] = metrics(M.proba(RV[v][lab], alpha), yr, labels=sorted(set(yr)))
    tr_res["rf_fuzzy"] = metrics(M.rf.predict_proba(RV["fuzzy"][lab]), yr, labels=sorted(set(yr)))
    tr_res["svm_fuzzy"] = metrics(M.svm.predict_proba(RV["fuzzy"][lab]), yr, labels=sorted(set(yr)))
    tr_res["coverage_fuzzy"] = metrics(M.cov(RV["fuzzy"][lab]), yr, labels=sorted(set(yr)))
    tr_res["tfidf_centroid"] = metrics(cosine_similarity(tf.transform(ures["text"].values[lab]), cent), yr, labels=sorted(set(yr)))
    for nm_, PP in (("fuzzy", M.proba(RV["fuzzy"][lab], alpha)), ("rf_fuzzy", M.rf.predict_proba(RV["fuzzy"][lab])),
                    ("tfidf_centroid", cosine_similarity(tf.transform(ures["text"].values[lab]), cent))):
        tr_res[nm_]["family"] = fam_acc(PP, yr)
    tr_res["family_majority"] = float(np.bincount([FAMILY[int(v)] for v in yr]).max() / len(yr))
    tr_res["n"] = len(lab); tr_res["classes"] = int(len(set(yr)))
    tr_res["majority"] = float(np.bincount(yr).max() / len(yr))
    res["transfer"] = tr_res

    #readiness validation vs matched_score
    JOB2ROLE = {"AI Engineer": "ML/AI Engineer", "Machine Learning (ML) Engineer": "ML/AI Engineer", "Intern (Generative AI Engineering - 2D/3D Image Generation)": "ML/AI Engineer",
                "Data Science Engineer": "Data Scientist", "Data Engineer": "Data Engineer", "DevOps Engineer": "DevOps/Cloud Engineer", "Database Administrator (DBA)": "Database Administrator",
                "Full Stack Developer (Python,React js)": "Full Stack Developer", "Senior iOS Engineer": "Mobile Developer", "Senior Software Engineer": "Software Engineer",
                "Network Support Engineer": "Network/Systems Admin",
                "System Administrator (Operation & Maintenance of Server, Storage & Service Desk System)": "Network/Systems Admin"}
    ridx = {n: i for i, n in enumerate(ROLE_NAMES)}
    rraw["jn"] = rraw["job_position_name"].str.strip()
    pr = rraw[rraw["jn"].isin(JOB2ROLE)].copy().reset_index(drop=True)
    pr["role_i"] = pr["jn"].map(lambda j: ridx[JOB2ROLE[j]])
    imp_all, _ = importance(st_all, y_all)               # full market database for candidate scoring
    imp_bin, _ = importance((st_all > 0).astype(float), y_all)
    imp_noidf, _ = importance(st_all, y_all, idf_on=False)
    X = Xr
    rd, rd_bin, rd_noidf = readiness(X, imp_all), readiness(X, imp_bin), readiness(X, imp_noidf)
    unw = (X @ (imp_all > 0).T) / (imp_all > 0).sum(1)
    cosr = cosine_similarity(X, imp_all)
    ri, ro = pr["rid"].values, pr["role_i"].values
    pr["rd"], pr["rd_bin"], pr["rd_noidf"] = rd[ri, ro], rd_bin[ri, ro], rd_noidf[ri, ro]
    pr["unw"], pr["cos"], pr["nsk"] = unw[ri, ro], cosr[ri, ro], X.sum(1)[ri]

    def req(s):
        if not isinstance(s, str): return set()
        o = set(K.extract_text(s, "alias"))
        for x in re.split(r"[\n,]", s):
            m = K.map_item(x, "fuzzy")
            if m: o.add(m)
        return o
    jobreq = {j: req(g["skills_required"].iloc[0]) for j, g in pr.groupby("jn")}
    pr["ov"] = [len({SKILLS[j] for j in np.where(X[a_] > 0)[0]} & jobreq[j]) / max(len(jobreq[j]), 1) if jobreq[j] else np.nan for a_, j in zip(pr["rid"], pr["jn"])]
    jt = {j: g["responsibilities.1"].fillna("").iloc[0] + " " + str(g["skills_required"].iloc[0]) for j, g in pr.groupby("jn")}
    tfv = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(list(ures["text"]) + list(jt.values()))
    pr["tfidf"] = [float(cosine_similarity(tfv.transform([ures["text"].iloc[a_]]), tfv.transform([jt[j]]))[0, 0]) for a_, j in zip(pr["rid"], pr["jn"])]

    def sp(col, d=pr):
        dd = d.dropna(subset=[col]); return float(spearmanr(dd[col], dd["matched_score"])[0])
    scorers = {"Job-ad skill overlap": "ov", "TF-IDF cosine (resume vs job ad)": "tfidf", "Unweighted coverage": "unw", "Cosine to role profile": "cos",
               "Readiness, no specificity (idf off)": "rd_noidf", "Readiness, binary mentions": "rd_bin", "Readiness (proposed)": "rd"}
    pooled = {k: sp(c) for k, c in scorers.items()}

    def within(col):
        out, top1, ndcg = [], [], []
        for _, g in pr.groupby("rid"):
            g = g.dropna(subset=[col])
            if len(g) < 4 or g[col].nunique() < 2: continue
            out.append(spearmanr(g[col], g["matched_score"])[0])
            top1.append(g["matched_score"].iloc[int(np.argmax(g[col].values))] >= np.sort(g["matched_score"].values)[-3])
            rel = g["matched_score"].values; rk = np.argsort(-g[col].values)[:3]
            dcg = sum(rel[j] / np.log2(i + 2) for i, j in enumerate(rk)); idcg = sum(v / np.log2(i + 2) for i, v in enumerate(np.sort(rel)[::-1][:3]))
            ndcg.append(dcg / idcg)
        return dict(spearman=float(np.nanmean(out)), top1_in_top3=float(np.mean(top1)), ndcg3=float(np.mean(ndcg)))
    wres = {k: within(c) for k, c in scorers.items()}
    per_job = {j: float(spearmanr(g["rd"], g["matched_score"])[0]) for j, g in pr.groupby("jn")}
    res["validation"] = dict(pairs=len(pr), jobs=int(pr["jn"].nunique()), resumes=int(pr["rid"].nunique()), pooled=pooled, within=wres, per_job=per_job,
                             jobs_with_required_list=int(sum(1 for v in jobreq.values() if v)))
    # supervised regression of matched_score with grouped CV
    feats = ["rd", "unw", "cos", "tfidf", "nsk"]
    F = pr[feats].values; yy = pr["matched_score"].values; grp = pr["rid"].values
    Fj = np.column_stack([F, pd.get_dummies(pr["jn"]).values.astype(float)])
    reg = {"Ridge": lambda: Ridge(1.0), "Random Forest": lambda: RandomForestRegressor(200, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
           "SVR (RBF)": lambda: SVR(C=1.0, epsilon=0.05)}
    regres = {}
    for nm, mk in reg.items():
        for tag, FF in ((nm, F), (nm + " + job id", Fj)):
            pred = np.zeros(len(yy))
            for a_, b_ in GroupKFold(5).split(FF, yy, grp):
                sc = StandardScaler().fit(FF[a_]); mm = mk().fit(sc.transform(FF[a_]), yy[a_]); pred[b_] = mm.predict(sc.transform(FF[b_]))
            regres[tag] = dict(r2=float(r2_score(yy, pred)), mae=float(mean_absolute_error(yy, pred)), rho=float(spearmanr(pred, yy)[0]))
    regres["Mean baseline"] = dict(r2=0.0, mae=float(np.abs(yy - yy.mean()).mean()), rho=0.0)
    res["regression"] = regres
    pr[["rid", "jn", "role_i", "matched_score", "rd", "unw", "cos", "tfidf", "nsk"]].to_csv(os.path.join(a.out, "pairs_data.csv"), index=False)

    #skill recovery (gap) evaluation on held-out postings
    pop = (st_all > 0).mean(0)
    rr = np.random.default_rng(5)
    hit = {m: [] for m in ("importance", "frequency (no idf)", "popularity", "random")}
    for i in te:
        have = np.where(Xb[i] > 0)[0]
        k = y[i]
        cand_have = [j for j in have if imp[k, j] > 0]
        if len(cand_have) < 4: continue
        hid = rr.choice(cand_have, max(1, int(round(0.3 * len(cand_have)))), replace=False)
        vis = set(have) - set(hid)
        pool_ = [j for j in np.where(imp[k] > 0)[0] if j not in vis]
        if len(pool_) < 5: continue
        for m in hit:
            if m == "importance": o = sorted(pool_, key=lambda j: -imp[k, j])
            elif m == "frequency (no idf)": o = sorted(pool_, key=lambda j: -freq[k, j])
            elif m == "popularity": o = sorted(pool_, key=lambda j: -pop[j])
            else: o = list(rr.permutation(pool_))
            hit[m].append(len(set(o[:5]) & set(hid)) / len(hid))
    res["skill_recovery"] = {m: float(np.mean(v)) for m, v in hit.items()}
    res["skill_recovery"]["n"] = len(hit["random"])

    #importance-variant ablation: coverage-only role accuracy, hidden-skill recovery, matched_score correlation
    abl = {}
    variants = {"Proposed (freq x idf, required/preferred)": (S_[tr], True), "No specificity (freq only)": (S_[tr], False),
                "Binary mentions (no required/preferred)": ((S_[tr] > 0).astype(float), True)}
    for nm, (strn, idf_on) in variants.items():
        iv, fv = importance(strn, y[tr], idf_on=idf_on)
        acc_c = float(accuracy_score(y[te], readiness(Xb[te], iv).argmax(1)))
        hits = []
        rr2 = np.random.default_rng(5)
        for i in te:
            have = np.where(Xb[i] > 0)[0]; k = y[i]
            ch = [j for j in have if iv[k, j] > 0]
            if len(ch) < 4: continue
            hid = rr2.choice(ch, max(1, int(round(0.3 * len(ch)))), replace=False)
            vis = set(have) - set(hid); pool_ = [j for j in np.where(iv[k] > 0)[0] if j not in vis]
            if len(pool_) < 5: continue
            o = sorted(pool_, key=lambda j: -iv[k, j]); hits.append(len(set(o[:5]) & set(hid)) / len(hid))
        abl[nm] = dict(cov_acc=acc_c, recovery=float(np.mean(hits)), n_req=float(np.mean([(iv[k] > 0).sum() for k in range(R_N)])))
    rho_of = {"Proposed (freq x idf, required/preferred)": pooled["Readiness (proposed)"], "No specificity (freq only)": pooled["Readiness, no specificity (idf off)"],
              "Binary mentions (no required/preferred)": pooled["Readiness, binary mentions"]}
    for nm in abl: abl[nm]["rho"] = rho_of[nm]
    res["importance_ablation"] = abl

    #roadmap on real candidates
    budgets = list(range(0, 401, 20))
    rm = {m: np.zeros((len(lab), len(budgets))) for m in ("proposed", "importance", "popularity", "random")}
    rg = np.random.default_rng(11)
    base = np.array([readiness(X[i:i + 1], imp_all)[0, ro_] for i, ro_ in zip(ures["rid"].values[lab], yr)])
    for j, (i, k) in enumerate(zip(lab, yr)):
        held = [SKILLS[q] for q in np.where(X[i] > 0)[0]]
        for m in rm:
            seq = plan(held, k, imp_all, "greedy" if m == "proposed" else m, rg, pop)
            for bi, b in enumerate(budgets):
                rm[m][j, bi] = gain_at(X[i], seq, k, imp_all, b)
    res["roadmap"] = {m: dict(g100=float(rm[m][:, budgets.index(100)].mean() * 100), g200=float(rm[m][:, budgets.index(200)].mean() * 100),
                              g300=float(rm[m][:, budgets.index(300)].mean() * 100),
                              reach50=float(np.mean((base + rm[m][:, budgets.index(400)]) >= 0.50) * 100))
                      for m in rm}
    res["readiness_before"] = dict(mean=float(base.mean() * 100), median=float(np.median(base) * 100), n=len(lab))

    #importance stability (market refresh evidence)
    stab = {}
    r3 = np.random.default_rng(3)
    for n_ in (25, 50, 100, 200, 400):
        vals = []
        for rep in range(5):
            sub = np.concatenate([r3.permutation(np.where(y_all == k)[0])[:n_] for k in range(R_N)])
            imp_s, _ = importance(st_all[sub], y_all[sub])
            vals.append(np.mean([spearmanr(imp_s[k][imp_all[k] > 0], imp_all[k][imp_all[k] > 0])[0] for k in range(R_N)]))
        stab[n_] = float(np.mean(vals))
    res["stability"] = stab
    lt = post["listed_time"].values
    o2 = np.argsort(lt); h2 = len(o2) // 2
    imp_e, f_e = importance(st_all[o2[:h2]], y_all[o2[:h2]]); imp_l, f_l = importance(st_all[o2[h2:]], y_all[o2[h2:]])
    res["temporal_importance"] = dict(
        mean_rho=float(np.mean([spearmanr(imp_e[k][(imp_e[k] > 0) | (imp_l[k] > 0)], imp_l[k][(imp_e[k] > 0) | (imp_l[k] > 0)])[0] for k in range(R_N)])),
        top10_overlap=float(np.mean([len(set(np.argsort(-imp_e[k])[:10]) & set(np.argsort(-imp_l[k])[:10])) / 10 for k in range(R_N)])),
        biggest_rise=[(SKILLS[j], round(float((st_all[o2[h2:]][:, j] > 0).mean() - (st_all[o2[:h2]][:, j] > 0).mean()) * 100, 2))
                      for j in np.argsort(-((st_all[o2[h2:]] > 0).mean(0) - (st_all[o2[:h2]] > 0).mean(0)))[:3]])

    #salary analysis (Python twin of the R script)
    sal = post[post["normalized_salary"].between(25000, 400000)].copy()
    sidx = sal.index.values
    cnt = (st_all[sidx] > 0).sum(0)
    keep_sk = [j for j in np.argsort(-cnt) if cnt[j] >= 120][:30]
    Xs = (st_all[sidx][:, keep_sk] > 0).astype(float)
    role_d = pd.get_dummies(sal["role"].map(lambda i: ROLE_NAMES[i])).drop(columns=["Software Engineer"]).astype(float)
    lvl = sal["formatted_experience_level"].fillna("Unknown")
    lvl_d = pd.get_dummies(lvl).drop(columns=["Unknown"] if "Unknown" in lvl.values else lvl.unique()[:1]).astype(float)
    Z = np.column_stack([np.ones(len(sal)), Xs, role_d.values, lvl_d.values])
    ly = np.log(sal["normalized_salary"].values)
    beta, *_ = np.linalg.lstsq(Z, ly, rcond=None)
    resid = ly - Z @ beta; dof = len(ly) - Z.shape[1]
    cov_b = (resid @ resid / dof) * np.linalg.pinv(Z.T @ Z)
    se = np.sqrt(np.diag(cov_b))
    sk_names = [SKILLS[j] for j in keep_sk]
    coef = pd.DataFrame(dict(skill=sk_names, beta=beta[1:1 + len(keep_sk)], se=se[1:1 + len(keep_sk)], n=cnt[keep_sk]))
    coef["pct"] = (np.exp(coef["beta"]) - 1) * 100
    coef["t"] = coef["beta"] / coef["se"]
    r2s = 1 - (resid @ resid) / (((ly - ly.mean()) ** 2).sum())
    res["salary"] = dict(n=len(sal), r2=float(r2s), median=float(sal["normalized_salary"].median()),
                         top=coef.sort_values("pct", ascending=False).head(5)[["skill", "pct", "t"]].round(2).values.tolist(),
                         bottom=coef.sort_values("pct").head(3)[["skill", "pct", "t"]].round(2).values.tolist(),
                         n_sig=int((coef["t"].abs() > 1.96).sum()), n_sk=len(coef))
    ex = pd.DataFrame(Xs, columns=sk_names); ex.insert(0, "log_salary", ly); ex.insert(1, "role", sal["role"].map(lambda i: ROLE_NAMES[i]).values)
    ex.insert(2, "exp_level", lvl.values)
    ex.to_csv(os.path.join(a.out, "salary_data.csv"), index=False)
    coef.to_csv(os.path.join(a.out, "salary_coefficients_python.csv"), index=False)

    # ------------------------------------------------------------ figures
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.65))
    v = ["exact", "alias", "fuzzy"]; lbl = ["Exact\nname", "+ Alias\ndictionary", "+ Fuzzy\nmatch"]
    axs[0].bar(range(3), [mapped_t[x] * 100 for x in v], color=[C[5], C[0], C[2]])
    for i_, x in enumerate(v): axs[0].text(i_, mapped_t[x] * 100 + 1, f"{mapped_t[x] * 100:.1f}", ha="center", fontsize=5.5)
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(lbl, fontsize=5.3); axs[0].set_ylabel("IT-resume skill items mapped (%)"); axs[0].set_ylim(0, 100)
    axs[1].bar([0, 1], [ex_acc["exact"] * 100, ex_acc["alias"] * 100], color=[C[5], C[0]], width=0.55)
    for i_, x in enumerate(["exact", "alias"]): axs[1].text(i_, ex_acc[x] * 100 + 1, f"{ex_acc[x] * 100:.1f}", ha="center", fontsize=5.5)
    axs[1].set_xticks([0, 1]); axs[1].set_xticklabels(["Exact\nname", "+ Alias\ndictionary"], fontsize=5.3); axs[1].set_ylabel("Role accuracy on postings (%)"); axs[1].set_ylim(0, 100)
    fig.tight_layout(); savefig(fig, a.out, "fig_extraction.png")

    cm = confusion_matrix(y[te], Pprop.argmax(1), normalize="true")
    fig, ax = plt.subplots(figsize=(3.3, 2.9)); im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(R_N)); ax.set_yticks(range(R_N)); ax.set_xticklabels(SHORT, rotation=60, ha="right"); ax.set_yticklabels(SHORT)
    for i_ in range(R_N):
        for j_ in range(R_N):
            if cm[i_, j_] >= 0.08: ax.text(j_, i_, f"{cm[i_, j_]:.2f}"[1:], ha="center", va="center", fontsize=3.8, color="white" if cm[i_, j_] > 0.5 else "black")
    ax.set_xlabel("Predicted role"); ax.set_ylabel("True role (from posting title)"); fig.colorbar(im, fraction=0.04, pad=0.02)
    savefig(fig, a.out, "fig_confusion.png")

    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.7))
    axs[0].plot(alphas, [res["alpha_sweep"][str(x)] * 100 for x in alphas], "-o", ms=2.5, color=C[0], lw=0.9)
    axs[0].axvline(alpha, ls="--", lw=0.6, color=C[1]); axs[0].set_xlabel(r"Blend weight $\alpha$"); axs[0].set_ylabel("Test accuracy (%)"); axs[0].grid(lw=0.3)
    for (k, vv), c, mk in zip(rob.items(), (C[2], C[3], C[4], C[0]), "svDo"):
        axs[1].plot(np.array(drops) * 100, np.array(vv) * 100, marker=mk, ms=2.5, lw=0.9, color=c, label=k.replace("Weighted coverage only", "Coverage only"))
    axs[1].set_xlabel("Skills dropped (%)"); axs[1].legend(frameon=False, fontsize=4.8); axs[1].grid(lw=0.3)
    fig.tight_layout(); savefig(fig, a.out, "fig_alpha_robust.png")

    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    ax.plot([l[0] for l in lc], [l[1] * 100 for l in lc], "-s", ms=3, lw=1, color=C[3], label="Random Forest")
    ax.plot([l[0] for l in lc], [l[2] * 100 for l in lc], "-o", ms=3, lw=1, color=C[0], label="Proposed hybrid")
    ax.set_xlabel("Training postings"); ax.set_ylabel("Test accuracy (%)"); ax.legend(frameon=False); ax.grid(lw=0.3)
    savefig(fig, a.out, "fig_learning.png")

    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9))
    axs[0].hexbin(pr["rd"] * 100, pr["matched_score"], gridsize=22, cmap="Blues", mincnt=1, linewidths=0.1)
    axs[0].set_xlabel("Readiness score"); axs[0].set_ylabel("Dataset matched_score")
    axs[0].set_title(f"Spearman {pooled['Readiness (proposed)']:.2f}", fontsize=6)
    names = list(scorers); vals = [pooled[k] for k in names]
    axs[1].barh(range(len(names)), vals, color=[C[5]] * 4 + [C[3]] * 2 + [C[0]])
    axs[1].set_yticks(range(len(names))); axs[1].set_yticklabels(["Job-ad overlap", "TF-IDF cosine", "Unweighted cov.", "Cosine profile", "No idf", "Binary", "Proposed"], fontsize=5)
    axs[1].set_xlabel("Spearman " + r"$\rho$"); axs[1].grid(axis="x", lw=0.3)
    fig.tight_layout(); savefig(fig, a.out, "fig_validation.png")

    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    lab_ = {"proposed": "Gain-per-hour greedy (proposed)", "importance": "Importance order", "popularity": "Market popularity", "random": "Random missing skill"}
    for m, c, ls in zip(rm, (C[0], C[2], C[3], C[5]), ("-", "--", "-.", ":")):
        ax.plot(budgets, rm[m].mean(0) * 100, ls, color=c, lw=1.1, label=lab_[m])
    ax.set_xlabel("Learning budget (hours)"); ax.set_ylabel("Readiness gain (points)"); ax.legend(frameon=False, loc="upper left"); ax.grid(lw=0.3)
    savefig(fig, a.out, "fig_roadmap.png")

    sel_roles = [1, 2, 3, 9, 8, 6]
    top_sk = []
    for k in sel_roles:
        for j in np.argsort(-imp[k])[:4]:
            if SKILLS[j] not in top_sk: top_sk.append(SKILLS[j])
    top_sk = top_sk[:18]
    H = np.array([[imp[k, SIDX[s]] for s in top_sk] for k in sel_roles])
    fig, ax = plt.subplots(figsize=(3.45, 1.9)); im = ax.imshow(H, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(top_sk))); ax.set_xticklabels(top_sk, rotation=70, ha="right", fontsize=5); ax.set_yticks(range(len(sel_roles)))
    ax.set_yticklabels([ROLE_NAMES[k] for k in sel_roles], fontsize=5.2); fig.colorbar(im, fraction=0.03, pad=0.01).set_label("Importance", fontsize=5)
    savefig(fig, a.out, "fig_importance_heatmap.png")

    cats = list(dict.fromkeys(CATEGORY.values()))
    def catcov(i, ri):
        req_ = [SKILLS[j] for j in np.where(imp_all[ri] > 0)[0]]; have = {SKILLS[j] for j in np.where(X[i] > 0)[0]}
        out = []
        for c in cats:
            rc = [s for s in req_ if CATEGORY[s] == c]; tot = sum(imp_all[ri, SIDX[s]] for s in rc)
            out.append(sum(imp_all[ri, SIDX[s]] for s in rc if s in have) / tot if tot > 0 else 0.0)
        return out
    cand = [(i, k) for i, k in zip(lab, yr) if 0.30 < readiness(X[i:i + 1], imp_all)[0, k] < 0.60 and X[i].sum() >= 8]
    good = [(i, k) for i, k in cand if M.proba(X[i:i + 1], alpha).argmax() == k]
    pick, pk = good[0] if good else cand[0]
    pvec = M.proba(X[pick:pick + 1], alpha)[0]; top = np.argsort(-pvec)[:5]; ri_ = top[0]
    cov = catcov(pick, ri_); ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False)
    fig = plt.figure(figsize=(3.45, 1.9)); ax = fig.add_subplot(121, polar=True)
    ax.plot(np.r_[ang, ang[0]], np.r_[cov, cov[0]], color=C[0], lw=1); ax.fill(np.r_[ang, ang[0]], np.r_[cov, cov[0]], color=C[0], alpha=0.3)
    ax.plot(np.r_[ang, ang[0]], np.ones(len(cats) + 1), color=C[1], lw=0.6, ls="--")
    ax.set_xticks(ang); ax.set_xticklabels([c.replace(" & ", "\n& ") for c in cats], fontsize=4.3); ax.set_yticks([0.5, 1]); ax.set_yticklabels(["", ""]); ax.set_ylim(0, 1.05)
    ax.set_title(f"Skill radar: {ROLE_NAMES[ri_]}", fontsize=5.8, pad=9)
    ax2 = fig.add_subplot(122); ax2.barh(range(5)[::-1], pvec[top], color=[C[0]] + [C[5]] * 4)
    ax2.set_yticks(range(5)[::-1]); ax2.set_yticklabels([ROLE_NAMES[j] for j in top], fontsize=5)
    rd_s = readiness(X[pick:pick + 1], imp_all)[0, ri_] * 100
    ax2.set_title(f"Ranked matches (readiness {rd_s:.0f}/100)", fontsize=5.8); ax2.set_xlabel("Hybrid score"); ax2.grid(axis="x", lw=0.3)
    fig.tight_layout(); savefig(fig, a.out, "fig_dashboard.png")
    held = [SKILLS[q] for q in np.where(X[pick] > 0)[0]]
    seq = plan(held, ri_, imp_all, "greedy", rg, pop)
    res["sample"] = dict(role=ROLE_NAMES[ri_], readiness=float(rd_s), held=held[:12], n_held=len(held), steps=[(s, hours_of(s), CATEGORY[s]) for s in seq[:7]])

    ss = coef.sort_values("pct"); sel = pd.concat([ss.head(4), ss.tail(8)])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.barh(range(len(sel)), sel["pct"], xerr=1.96 * sel["se"] * 100 * np.exp(sel["beta"]), color=[C[1] if p < 0 else C[0] for p in sel["pct"]], error_kw=dict(lw=0.5), capsize=1)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel["skill"], fontsize=5.2); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Salary difference when skill is mentioned (%)"); ax.grid(axis="x", lw=0.3)
    savefig(fig, a.out, "fig_salary.png")

    fig, ax = plt.subplots(figsize=(3.4, 1.5))
    ax.plot(list(stab), list(stab.values()), "-o", ms=3, color=C[0], lw=1); ax.set_xlabel("Postings per role used to estimate importance"); ax.set_ylabel("Spearman vs full corpus"); ax.grid(lw=0.3)
    ax.set_ylim(0.4, 1.0)
    savefig(fig, a.out, "fig_stability.png")

    res["runtime_total_seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)
    print(json.dumps({k: res[k] for k in ("data", "models", "cv", "alpha", "transfer", "validation", "regression", "skill_recovery", "roadmap", "salary", "temporal", "temporal_importance", "stability")}, indent=1, default=float)[:9000])


if __name__ == "__main__":
    main()
