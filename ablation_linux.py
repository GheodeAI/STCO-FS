"""
Ablation incremental sobre el CRO real (popSize=20, init only) para medir la
contribución de cada mejora a la velocidad de simulación.

Self-contained: sólo depende de los datos en Data/Paper/ y de PyCROSL.

Variantes:
  V0  baseline (cv=5, tol=1e-4, no pre-std, con extras, Njobs=1)
  V1  V0 + sin extras (B)
  V2  V1 + pre-estandarizado (G)
  V3  V2 + tol=1e-3 (D)
  V4  V3 + cv=3 (E)
  V5  V4 + Njobs=-1   (paralelismo individuo, procesos joblib)
  V6  V4 + n_jobs=-1 en cross_val (paralelismo de folds, hilos)
  V7  V4 + ambos paralelismos
"""
import os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random as pyrandom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from PyCROSL.CRO_SL import CRO_SL
from PyCROSL.SubstrateInt import SubstrateInt
from PyCROSL.AbsObjectiveFunc import AbsObjectiveFunc


# ---- Datos y supermatrix ----
DATA = './Data/Paper/'
pred = pd.read_csv(DATA + 'predictors_dataset.csv', index_col=0)
pred.index = pd.to_datetime(pred.index)
tgt = pd.read_csv(DATA + 'target.csv', index_col=0)
tgt.index = pd.to_datetime(tgt.index)

MAX_LAG, MAX_WINDOW = 180, 60
MAX_SHIFT = MAX_LAG + MAX_WINDOW
HORIZON = 0
NLEN = len(pred)
P = pred.shape[1]
F = P * (MAX_SHIFT - HORIZON)

idx_pred = pred.index[MAX_SHIFT:]
valid_idx = idx_pred.intersection(tgt.index)

blocks, col_index = [], {}
LUT = np.full((P, MAX_SHIFT + 1), -1, dtype=np.int64)
cid = 0
for var_i, col in enumerate(pred.columns):
    s = pred[col].to_numpy()
    for lag in range(1, MAX_SHIFT + 1 - HORIZON):
        blocks.append(s[MAX_SHIFT - lag: NLEN - lag].reshape(-1, 1))
        col_index[(var_i, lag)] = cid
        LUT[var_i, lag] = cid
        cid += 1
X_full = np.hstack(blocks).astype(np.float32)
pos = idx_pred.get_indexer(valid_idx)
X_full = X_full[pos, :]
y_full = tgt.reindex(valid_idx)['Target'].to_numpy().astype(np.float32)

train_mask = (valid_idx.year >= 1950) & (valid_idx.year <= 2010)
test_mask = (valid_idx.year > 2010)
Xtr_raw = X_full[train_mask]; Xte_raw = X_full[test_mask]
ytr = y_full[train_mask]; yte = y_full[test_mask]

mu = Xtr_raw.mean(axis=0)
sigma = Xtr_raw.std(axis=0); sigma[sigma == 0] = 1
Xtr_std = ((Xtr_raw - mu) / sigma).astype(np.float32)


def hostinfo():
    import platform
    nc = os.cpu_count()
    print(f"machine={platform.machine()} system={platform.system()} cpu_count={nc}", flush=True)
    try:
        # BLAS info si está disponible
        import numpy.distutils.system_info as si
        print(f"numpy={np.__version__}", flush=True)
    except Exception:
        print(f"numpy={np.__version__}", flush=True)


hostinfo()
print(f"P={P} F={F} n_train={Xtr_raw.shape[0]}", flush=True)


def cols_of(sol):
    win, lag, sel = sol[:P].astype(int), sol[P:2*P].astype(int), sol[2*P:3*P].astype(int)
    out = []
    for i in range(P):
        if sel[i] == 0 or win[i] <= 0: continue
        l0 = max(1, int(lag[i]))
        l1 = min(MAX_SHIFT + 1, int(lag[i]) + int(win[i]))
        if l0 < l1:
            cs = LUT[i, l0:l1]; cs = cs[cs >= 0]; out.extend(cs.tolist())
    return out


class CfgObj(AbsObjectiveFunc):
    def __init__(self, *, with_extras, pre_std, tol, cv, cv_n_jobs):
        sup = np.concatenate([np.full(P, 60), np.full(P, 180), np.full(P, 1)])
        inf = np.concatenate([np.full(P, 1), np.full(P, 0), np.full(P, 0)])
        super().__init__(3 * P, "min", sup, inf)
        self.with_extras = with_extras
        self.pre_std = pre_std
        self.tol = tol
        self.cv = cv
        self.cv_n_jobs = cv_n_jobs
        self.cache = {}

    def objective(self, solution):
        key = tuple(solution)
        if key in self.cache: return self.cache[key]
        cols = cols_of(np.asarray(solution))
        if not cols:
            self.cache[key] = 100000.0; return 100000.0
        if self.pre_std:
            Xs_tr = Xtr_std[:, cols]
        else:
            Xs_tr = (Xtr_raw[:, cols] - mu[cols]) / sigma[cols]
        score = cross_val_score(
            LogisticRegression(tol=self.tol),
            Xs_tr, ytr, cv=self.cv, scoring='f1',
            n_jobs=self.cv_n_jobs,
        ).mean()
        if self.with_extras:
            Xs_te = (Xte_raw[:, cols] - mu[cols]) / sigma[cols]
            clf = LogisticRegression(tol=self.tol)
            clf.fit(Xs_tr, ytr)
            _ = f1_score(clf.predict(Xs_te), yte)
        v = 1/score if score > 0 else 1e6
        self.cache[key] = v; return v

    def random_solution(self):
        return np.random.choice(self.sup_lim[0], self.input_size, replace=True)
    def repair_solution(self, solution):
        return np.clip(solution, self.inf_lim, self.sup_lim)


def make_operators():
    return [
        SubstrateInt('BLXalpha', {'F': 0.8}),
        SubstrateInt('Multipoint'),
        SubstrateInt('HS', {'F': 0.7, 'Cr': 0.8, 'Par': 0.2}),
        SubstrateInt('Xor'),
    ]

def make_params(pop, neval, njobs):
    return {
        'popSize': pop, 'rho': 0.6, 'Fb': 0.98, 'Fd': 0.2, 'Pd': 0.8,
        'k': 3, 'K': 20, 'group_subs': True,
        'stop_cond': 'Neval', 'time_limit': 6000.0,
        'Ngen': 1000, 'Neval': neval, 'fit_target': 1000,
        'verbose': False, 'v_timer': 1, 'Njobs': njobs,
        'dynamic': True, 'dyn_method': 'success', 'dyn_metric': 'avg',
        'dyn_steps': 10, 'prob_amp': 0.01,
    }


POP, NEVAL = 20, 20
print(f"\nConfig CRO: popSize={POP}, Neval={NEVAL} (solo init)", flush=True)

variants = [
    ("V0 baseline (cv=5, tol=1e-4, extras, sin pre-std, Njobs=1)",
        dict(with_extras=True,  pre_std=False, tol=1e-4, cv=5, cv_n_jobs=None), 1),
    ("V1 + sin extras (B)",
        dict(with_extras=False, pre_std=False, tol=1e-4, cv=5, cv_n_jobs=None), 1),
    ("V2 + pre-estandarizado (G)",
        dict(with_extras=False, pre_std=True,  tol=1e-4, cv=5, cv_n_jobs=None), 1),
    ("V3 + tol=1e-3 (D)",
        dict(with_extras=False, pre_std=True,  tol=1e-3, cv=5, cv_n_jobs=None), 1),
    ("V4 + cv=3 (E)",
        dict(with_extras=False, pre_std=True,  tol=1e-3, cv=3, cv_n_jobs=None), 1),
    ("V5 V4 + Njobs=-1 (procesos)",
        dict(with_extras=False, pre_std=True,  tol=1e-3, cv=3, cv_n_jobs=None), -1),
    ("V6 V4 + n_jobs=-1 en cross_val (hilos)",
        dict(with_extras=False, pre_std=True,  tol=1e-3, cv=3, cv_n_jobs=-1), 1),
    ("V7 V4 + AMBOS paralelismos",
        dict(with_extras=False, pre_std=True,  tol=1e-3, cv=3, cv_n_jobs=-1), -1),
]

print(f"\n{'variante':<58} {'time':>7}  {'speedup':>8}  {'best 1/F1':>10}", flush=True)
results = []
for label, cfg, njobs in variants:
    np.random.seed(2026); pyrandom.seed(2026)
    obj = CfgObj(**cfg)
    cro = CRO_SL(obj, make_operators(), make_params(POP, NEVAL, njobs))
    t0 = time.perf_counter()
    sol, fit = cro.optimize()
    t = time.perf_counter() - t0
    base = results[0][1] if results else t
    print(f"{label:<58} {t:>6.1f}s  x{base/t:>7.2f}  {fit:>10.4f}", flush=True)
    results.append((label, t, fit, sol))

sol_V0, sol_V4 = results[0][3], results[4][3]
same = np.array_equal(sol_V0, sol_V4)
print(f"\nMisma mejor solución V0 vs V4: {same}", flush=True)
