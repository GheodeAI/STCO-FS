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


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
import time

from joblib import parallel_backend



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

X_blocks = []
col_meta = []
col_index = {} # To identify lags
col_id = 0
for var_i, col in enumerate(pred_dataframe.columns):
    s = pred_dataframe[col].to_numpy() # each column is a time series
    for lag in range(1, MAX_SHIFT + 1 - HORIZON):
        xlag = s[MAX_SHIFT - lag : NLEN - lag]
        X_blocks.append(xlag.reshape(-1, 1))
        col_meta.append((var_i, lag))
        col_index[(var_i, lag)] = col_id 
        col_id += 1

X_full = np.hstack(X_blocks).astype(np.float32)
pos = idx_pred.get_indexer(valid_idx)
X_full = X_full[pos, :]

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

# Cache to avoid computing repeated solutions
fitness_cache = {}

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
        t0 = time.perf_counter()
        # print(solution)
        # Read data
        # sol_file = pd.read_csv(indiv_file,sep=' ',header=0)
        history = []
        key = tuple(solution)
        if key in fitness_cache: # If the solution has been computed before, return the cached value
            return fitness_cache[key]

        # BOTTLENECK!
        # # Create dataset according to solution
        # dataset_opt = target_dataset.copy()
        # for i,col in enumerate(pred_dataframe.columns):
        #     if variable_selection[i] == 0 or time_sequences[i] == 0:
        #         continue
        #     for j in range(time_sequences[i]):
        #         dataset_opt[str(col)+'_lag'+str(time_lags[i]+j)] = pred_dataframe[col].shift(time_lags[i]+j)

        selected_cols = solution_to_selected_cols(
            solution, 
            pred_dataframe.shape[1], 
            col_index, 
            MAX_SHIFT
            )
        if len(selected_cols) == 0:
            return 100000

        X_train = X_train_full[:, selected_cols]
        Y_train = y_train_full

        X_test = X_test_full[:, selected_cols]
        Y_test = y_test_full

        X_std_train = (X_train - mu[selected_cols]) / sigma[selected_cols]
        X_std_test = (X_test - mu[selected_cols]) / sigma[selected_cols]



        # Train model
        clf = LogisticRegression(class_weight='balanced')
        # Apply cross validation
        # clf.fit(X_std_train, Y_train)
        score = cross_val_score(clf, X_std_train, Y_train, cv=5, scoring="f1").mean()

        # Save solution
        history.append([score, Y_test, solution])
        clf.fit(X_std_train, Y_train)
        Y_pred = clf.predict(X_std_test)
        print(score, f1_score(Y_pred,Y_test))

        # Guard against score == 0 (frequent with imbalanced target): 1/0 -> inf
        # would collapse many solutions to the same fitness and break the
        # dynamic (avg-based) operator adaptation.
        fitness = 1/score if score > 0 else 100000
        fitness_cache[key] = fitness
        elapsed = time.perf_counter() - t0
        print(f"objective time: {elapsed:.4f} s")

        return fitness
    
    """
    This will be the function used to generate random vectorsfor the initializatio of the algorithm
    """
    def random_solution(self):
        return np.random.choice(self.sup_lim[0], self.size, replace=True)
    
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
    "Neval": 15000,
    "fit_target": 1000,

    "verbose": True,
    "v_timer": 1,
    "Njobs": 1,

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
    SubstrateInt("Xor"),
]

cro_alg = CRO_SL(objfunc, operators, params)

solution, obj_value = cro_alg.optimize()

solution.tofile(path_output+solution_file, sep=',')

