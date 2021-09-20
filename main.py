# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np

# This is a test



from seaborn import heatmap
import matplotlib.pyplot as plt
import scipy.stats as st
from optuna.integration import LightGBMPruningCallback
pd.set_option('expand_frame_repr', False)
from scipy.stats import randint, uniform
from sklearn import decomposition
from sklearn import datasets
from sklearn.metrics import accuracy_score, roc_auc_score
import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from collections import namedtuple
import optuna



def objective(trial,x,y):
    xtrain, xval = x
    ytrain, yval = y

    fixed_para = {
        "early_stopping_rounds": 5,
        'eval_metric': 'auc',
        'eval_set': [(xval, yval)],
        # 'eval_names': ['val set'],
        'verbose': 25,
        'callbacks': [LightGBMPruningCallback(trial, 'auc')],
    }

    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        # "boosting_type": trial.suggest_categorical("boosting_type",[ 'goss', 'gbdt']),
        "boosting_type": 'gbdt',
        "n_estimators":trial.suggest_int("n_estimators", 100, 1500, log=True),
        # "max_depth":trial.suggest_int("max_depth", 1, 8, log=True),
        # "learning_rate": trial.suggest_float("learning_rate", 0.01, 1., log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 5., log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8,5., log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    if param["boosting_type"] in ['rf','gbdt' ]:
        param["feature_fraction"]= trial.suggest_float("feature_fraction", 0.4, 1.0)
        param["bagging_fraction"]= trial.suggest_float("bagging_fraction", 0.4, 1.0)
        param["bagging_freq"]= trial.suggest_int("bagging_freq", 1, 7)


    gbm = LGBMClassifier(**param)
    gbm.fit(xtrain,ytrain,**fixed_para)
    preds = gbm.predict_proba(xval)[:,1]

    return roc_auc_score(yval, preds)

def optuna_():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    X = pd.DataFrame(train.drop(columns=["claim", "id"]))
    xtest = pd.DataFrame(test.drop(columns=["id"]))
    Y = train["claim"]
    folds = 10

    x, xval2, y, yval2 = ms.train_test_split(X, Y, test_size=.1, shuffle=True, random_state=0)
    xtrain, xval, ytrain, yval = ms.train_test_split(x, y, test_size=.2, shuffle=True, random_state=0)
    # xval, xtest, yval, ytest = ms.train_test_split(xval_, yval_, test_size=.5, shuffle=True, random_state=0)


    study = optuna.create_study(direction='maximize', study_name="robin")
    study.optimize(lambda i : objective(i, (xtrain, xval), (ytrain, yval)), n_trials=200)
    print()
    print(f"best trial: {([j for i, j in study.best_trial.intermediate_values.items()])[-1]}")
    print(f"Trial #{study.best_trial.number} best trial: {study.best_trial.params}")

    # holding my scores
    score = np.zeros((1, folds))
    # holding my weights for weighted average
    weights = np.zeros((1, folds))
    # pred score from test data
    predictions = np.zeros((len(xtest), folds))
    df_split = ms.StratifiedKFold(n_splits=folds, shuffle=True)

    model = LGBMClassifier(n_jobs = -1, **study.best_trial.params)

    for counter, (trn, val) in enumerate(df_split.split(x, y)):
        model.fit(x.iloc[trn,:], y.iloc[trn])
        w = model.predict_proba(xval2)[:,1]
        score[0,counter]= roc_auc_score(yval2, w)
        predictions[:,counter] = model.predict_proba(xtest.values)[:,1]
        if counter == folds -1:
            score = np.reshape(np.dot(1/np.sum(score), score), (folds, 1))
            print(score)


    final = pd.DataFrame(test["id"])
    final = final.merge(pd.DataFrame(np.dot(predictions, score)), right_index=True, left_index=True)
    final.columns = ["id", "claim"]
    print(final.head(5))


    final.to_csv("sub_v6.csv", index=False)




def fill_na(X, X_):
    for i in X.columns:
        X_[i] = X_[i].fillna(np.mean(X.loc[X[i].isna() == False, i]))
    return X_




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # main()
    optuna_()
