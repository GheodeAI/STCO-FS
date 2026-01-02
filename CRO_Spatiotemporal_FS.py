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
NLEN = len(pred_dataframe)
lagged_pred_dataframe = pred_dataframe[MAX_LAG:]

# Split the dataset into train and test
first_train = 1951
last_train = 2010

train_indices = (target_dataset.index.year >=first_train) & (target_dataset.index.year <=last_train)
test_indices = (target_dataset.index.year >last_train)

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
        # print(solution)
        # Read data
        sol_file = pd.read_csv(indiv_file,sep=' ',header=0)

        # Read solution
        time_sequences = np.append(np.array(solution[:pred_dataframe.shape[1]]).astype(int),1)
        time_lags = np.append(np.array(solution[pred_dataframe.shape[1]:(2*pred_dataframe.shape[1])]).astype(int),1)
        variable_selection = np.array(solution[(2*pred_dataframe.shape[1]):]).astype(int)

        if sum(variable_selection) == 0:  # If no variables are selected, return a high value
            return 100000


        # BOTTLENECK!
        # # Create dataset according to solution
        # dataset_opt = target_dataset.copy()
        # for i,col in enumerate(pred_dataframe.columns):
        #     if variable_selection[i] == 0 or time_sequences[i] == 0:
        #         continue
        #     for j in range(time_sequences[i]):
        #         dataset_opt[str(col)+'_lag'+str(time_lags[i]+j)] = pred_dataframe[col].shift(time_lags[i]+j)

        X_blocks = []
        col_meta = []
        for var_i, col in enumerate(pred_dataframe.columns):
            s = pred_dataframe[col].to_numpy()
            for lag in range(1, MAX_LAG + 1):
                xlag = s[MAX_LAG - lag : NLEN - lag]
        # Split dataset into train and test

        train_dataset = dataset_opt[train_indices]
        test_dataset = dataset_opt[test_indices]
        
        # Standardize data
        Y_column = 'Target' 
            
        X_train=train_dataset[train_dataset.columns.drop([Y_column]) ]
        Y_train=train_dataset[Y_column]

        X_test=test_dataset[test_dataset.columns.drop([Y_column]) ]
        Y_test=test_dataset[Y_column]
            
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        X_std_train = scaler.fit(X_train)

        X_std_train = scaler.transform(X_train)
        X_std_test = scaler.transform(X_test)

        X_train=pd.DataFrame(X_std_train,columns=X_train.columns,index=X_train.index)
        X_test=pd.DataFrame(X_std_test,columns=X_test.columns,index=X_test.index)


        # Train model
        from sklearn.metrics import f1_score, mean_absolute_error
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()


        # Apply cross validation
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(clf, X_train, Y_train, cv=5,scoring='f1')
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print(score.mean(), f1_score(Y_pred,Y_test))

        # Save solution
        sol_file = pd.concat([sol_file, pd.DataFrame({'CV': [score.mean()], 'Test': [f1_score(Y_pred, Y_test)], 'Sol': [solution]})], ignore_index=True)
        sol_file.to_csv(indiv_file,sep=' ',header=sol_file.columns,index=None)
        return 1/score.mean()
    

    
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

