from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateReal import *
from PyCROSL.SubstrateInt import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import random


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
import time
from threadpoolctl import threadpool_limits

from cro_utils import (
    ConcurrentMemo,
    flip_driver_selection,
    sample_bounded_integers,
    threaded_cross_val_score,
)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


CRO_VERBOSE = env_flag("CRO_VERBOSE", default=True)
CRO_RANDOM_SEED = os.environ.get("CRO_SEED")
CRO_RANDOM_SEED = int(CRO_RANDOM_SEED) if CRO_RANDOM_SEED is not None else None


"""
File names to store the solutions provided by the algorithm
"""

filename = 'Test_Paper_1'
path_output = './Results/Test_Paper/'
# Create directory
if not os.path.exists(path_output):
    os.makedirs(path_output)

"""
Path and name of the predictor dataset and target dataset
"""
path_input = './Data/Paper/'
predictor_file = 'predictors_dataset.csv'
target_file = 'target.csv'



# Load the dataset
pred_dataframe = pd.read_csv(path_input+predictor_file, index_col=0)
pred_dataframe.index = pd.to_datetime(pred_dataframe.index)
target_dataset = pd.read_csv(path_input+target_file, index_col=0)
target_dataset.index = pd.to_datetime(target_dataset.index)

# The positional lagging below assumes the predictors are daily and gap-free:
# moving back 'lag' positions must equal moving back 'lag' calendar days. If the
# source is ever regenerated with a missing day (or duplicates/out-of-order
# dates), the lags would silently misalign, so fail loudly here instead.
_expected_idx = pd.date_range(pred_dataframe.index.min(), pred_dataframe.index.max(), freq='D')
if not pred_dataframe.index.equals(_expected_idx):
    raise ValueError(
        "Predictors index must be daily, ordered and gap-free for positional "
        f"lagging: got {len(pred_dataframe)} rows vs {len(_expected_idx)} "
        "expected daily dates between min and max."
    )


# Create an empty file to store the solutions provided by the algorithm
sol_data = pd.DataFrame(columns=['CV','Test','Sol'])

indiv_file = path_output+filename+'.csv'
solution_file = 'CRO_LogReg_'+filename+'.csv'
sol_data.to_csv(indiv_file,sep=' ',header=sol_data.columns,index=None)


# LAG and WINDOW
MAX_LAG = 180
MAX_WINDOW = 60
MAX_SHIFT = MAX_LAG + MAX_WINDOW
HORIZON = 0
NLEN = len(pred_dataframe)
idx_pred = pred_dataframe.index[MAX_SHIFT:]
valid_idx = idx_pred.intersection(target_dataset.index) # Valid indices

# Preallocate the supermatrix and fill it column by column directly in
# float32, keeping only the valid rows. Avoids building a list of ~17k
# blocks + hstack + float64 cast, which peaked at ~5.4 GB; this stays ~0.6 GB.
col_meta = []
col_index = {} # To identify lags
n_lags = MAX_SHIFT - HORIZON
pos = idx_pred.get_indexer(valid_idx) # valid rows (intersection with target)
X_full = np.empty((len(valid_idx), pred_dataframe.shape[1] * n_lags), dtype=np.float32)
col_id = 0
for var_i, col in enumerate(pred_dataframe.columns):
    s = pred_dataframe[col].to_numpy() # each column is a time series
    for lag in range(1, n_lags + 1):
        X_full[:, col_id] = s[MAX_SHIFT - lag : NLEN - lag][pos]
        col_meta.append((var_i, lag))
        col_index[(var_i, lag)] = col_id
        col_id += 1

y_full = target_dataset.reindex(valid_idx)['Target'].to_numpy()

# Train and test masks
first_train_year = 1950
last_train_year = 2010
train_mask = ((valid_idx.year >= first_train_year) & (valid_idx.year <= last_train_year))
test_mask = ((valid_idx.year > last_train_year))

# First split to avoid overload in memory
X_train_full = X_full[train_mask]
X_test_full  = X_full[test_mask]
y_train_full = y_full[train_mask]
y_test_full  = y_full[test_mask]

# Avoiding standardization per generation
mu = X_train_full.mean(axis=0)
sigma = X_train_full.std(axis=0)
sigma[sigma == 0] = 1

LOGREG_PARAMS = {"class_weight": "balanced"}
CV_FOLDS = 5
# Parallelism policy:
#   Recommended: CRO_N_JOBS=1 and CRO_CV_N_JOBS=5. The CRO remains
#   sequential, while the five CV folds of each fitness evaluation run in
#   shared-memory threads. This preserves the CRO trajectory and gave the
#   best measured runtime/memory trade-off on the real dataset.
#   Experimental: CRO_N_JOBS>1 evaluates larvae from one generation in
#   parallel. In that mode CV_N_JOBS is forced to 1 so parallelism is not
#   nested; this mode uses more memory and is not the default.
# CRO_NEVAL controls the search budget and can change the selected solution;
# the two job-count parameters only change how independent work is scheduled.
CRO_N_JOBS = int(os.environ.get("CRO_N_JOBS", "1"))
if CRO_N_JOBS < 1:
    raise ValueError("CRO_N_JOBS must be a positive integer.")
REQUESTED_CV_N_JOBS = min(
    CV_FOLDS,
    max(1, int(os.environ.get("CRO_CV_N_JOBS", os.cpu_count() or 1))),
)
CV_N_JOBS = 1 if CRO_N_JOBS != 1 else REQUESTED_CV_N_JOBS
CV_BLAS_THREADS_PER_JOB = 1
CRO_NEVAL = int(os.environ.get("CRO_NEVAL", "15000"))
if CRO_NEVAL < 1:
    raise ValueError("CRO_NEVAL must be positive.")

def solution_to_selected_cols(solution, p, col_index, max_shift):
    time_sequences = np.array(solution[:p]).astype(int)
    time_lags      = np.array(solution[p:2*p]).astype(int)
    variable_sel   = np.array(solution[2*p:3*p]).astype(int)

    selected_cols = []
    for i in range(p):
        if variable_sel[i] == 0:
            continue
        win = int(time_sequences[i])
        if win <= 0:
            continue
        start = int(time_lags[i])
        for j in range(win):
            lag = start + j
            if 1 <= lag <= max_shift:
                selected_cols.append(col_index[(i, lag)])
    return selected_cols

"""
All the following methods will have to be implemented for the algorithm to work properly
with the same inputs, except for the constructor 
"""
class ml_prediction(AbsObjectiveFunc):
    """
    This is the constructor of the class, here is where the objective function can be setted up.
    In this case we will only add the size of the vector as a parameter.
    """
    def __init__(self, size):
        self.size = size
        self.opt = "min" # it can be "max" or "min"
        self.fitness_cache = ConcurrentMemo()

        # We set the limits of the vector (window size, time lags and variable selection)
        self.sup_lim = np.append(np.append(np.repeat(60, pred_dataframe.shape[1]),np.repeat(180, pred_dataframe.shape[1])),np.repeat(1, pred_dataframe.shape[1]))  # array where each component indicates the maximum value of the component of the vector
        self.inf_lim = np.append(np.append(np.repeat(1, pred_dataframe.shape[1]),np.repeat(0, pred_dataframe.shape[1])),np.repeat(0, pred_dataframe.shape[1])) # array where each component indicates the minimum value of the component of the vector
        # we call the constructor of the superclass with the size of the vector
        # and wether we want to maximize or minimize the function 
        super().__init__(self.size, self.opt, self.sup_lim, self.inf_lim)
    
    """
    This will be the objective function, that will recieve a vector and output a number
    """
    def objective(self, solution):
        selected_cols = solution_to_selected_cols(
            solution, 
            pred_dataframe.shape[1], 
            col_index, 
            MAX_SHIFT
            )
        # uint16 is sufficient for the 17,040 possible driver-lag columns and
        # gives equivalent phenotypes one compact shared cache key.
        if selected_cols and max(selected_cols) > np.iinfo(np.uint16).max:
            raise ValueError("Selected column index does not fit in uint16.")
        key = np.asarray(selected_cols, dtype=np.uint16).tobytes()

        def compute_fitness():
            t0 = time.perf_counter()
            if len(selected_cols) == 0:
                return 100000

            X_train = X_train_full[:, selected_cols]
            X_std_train = (X_train - mu[selected_cols]) / sigma[selected_cols]

            clf = LogisticRegression(**LOGREG_PARAMS)
            scores = threaded_cross_val_score(
                clf,
                X_std_train,
                y_train_full,
                cv=CV_FOLDS,
                scoring="f1",
                n_jobs=CV_N_JOBS,
                blas_threads=CV_BLAS_THREADS_PER_JOB,
                # BLAS is limited once around the whole CRO run. Per-objective
                # contexts would race when several candidates run in threads.
                limit_blas=False,
            )
            score = scores.mean()

            # Guard against score == 0 (frequent with imbalanced target): 1/0
            # would collapse solutions and break avg-based operator adaptation.
            fitness = 1/score if score > 0 else 100000
            elapsed = time.perf_counter() - t0
            print(f"CV f1: {score}  objective time: {elapsed:.4f} s")
            return fitness

        return self.fitness_cache.get_or_compute(key, compute_fitness)
    
    """
    This will be the function used to generate random vectorsfor the initializatio of the algorithm
    """
    def random_solution(self):
        return sample_bounded_integers(self.inf_lim, self.sup_lim)
    
    """
    This will be the function that will repair solutions, or in other words, makes a solution
    outside the domain of the function into a valid one.
    If this is not needed simply return "solution"
    """
    def repair_solution(self, solution):

        # unique = np.unique(solution)
        # if len(unique) < len(solution):
        #     pool = np.setdiff1d(np.arange(self.inf_lim[0], self.sup_lim[0]), unique)
        #     new = np.random.choice(pool, len(solution) - len(unique), replace=False)
        #     solution = np.concatenate((unique, new))
        return np.clip(solution, self.inf_lim, self.sup_lim)
objfunc = ml_prediction(3*pred_dataframe.shape[1])

params = {
    "popSize": 100,
    "rho": 0.6,
    "Fb": 0.98,
    "Fd": 0.2,
    "Pd": 0.8,
    "k": 3,
    "K": 20,
    "group_subs": True,

    "stop_cond": "Neval",
    "time_limit": 4000.0,
    "Ngen": 10000,
    "Neval": CRO_NEVAL,
    "fit_target": 1000,

    "verbose": CRO_VERBOSE,
    "v_timer": 1,
    # Only one level is parallel at once: candidates when CRO_N_JOBS != 1,
    # otherwise CV folds. Both modes share the read-only feature matrix.
    "Njobs": CRO_N_JOBS,

    "dynamic": True,
    "dyn_method": "success",
    "dyn_metric": "avg",
    "dyn_steps": 10,
    "prob_amp": 0.01,

    # "prob_file": "prob_history_"+filename+".csv",
    # "popul_file": "last_population"+filename+".csv",
    # "history_file": "fit_history_"+filename+".csv",
    "solution_file": "best_solution_"+filename+".csv",
    # "indiv_file": "indiv_hisotry_"+filename+".csv",
}

operators = [
    SubstrateInt("BLXalpha", {"F":0.8}),
    SubstrateInt("Multipoint"),
    SubstrateInt("HS", {"F": 0.7, "Cr":0.8,"Par":0.2}),
    # Byte-wise XOR made 0 -> 1 overwhelmingly more likely in the binary block.
    # This custom mutation flips only driver-selection bits, symmetrically.
    SubstrateInt("Custom", {"function": flip_driver_selection, "N": 5}),
]

cro_alg = CRO_SL(objfunc, operators, params)

if CRO_RANDOM_SEED is not None:
    random.seed(CRO_RANDOM_SEED)
    np.random.seed(CRO_RANDOM_SEED)

print(
    f"Parallelism: CRO workers={CRO_N_JOBS}, CV workers={CV_N_JOBS}, "
    f"BLAS threads={CV_BLAS_THREADS_PER_JOB}"
)
with threadpool_limits(limits=CV_BLAS_THREADS_PER_JOB):
    solution, obj_value = cro_alg.optimize()

solution.tofile(path_output+solution_file, sep=',')

# Final evaluation of the best solution on the held-out TEST set, done once.
best_cols = solution_to_selected_cols(
    solution, pred_dataframe.shape[1], col_index, MAX_SHIFT
)
if len(best_cols) > 0:
    X_std_train = (X_train_full[:, best_cols] - mu[best_cols]) / sigma[best_cols]
    X_std_test = (X_test_full[:, best_cols] - mu[best_cols]) / sigma[best_cols]
    clf = LogisticRegression(**LOGREG_PARAMS)
    clf.fit(X_std_train, y_train_full)
    y_pred = clf.predict(X_std_test)
    print(
        f"Best solution: CV fitness={obj_value:.4f}  #cols={len(best_cols)}  "
        f"test f1={f1_score(y_test_full, y_pred):.4f}"
    )
else:
    print("Best solution selects no columns.")
