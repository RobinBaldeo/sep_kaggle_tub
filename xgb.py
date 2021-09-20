# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
pd.set_option('expand_frame_repr', False)
from sklearn.metrics import accuracy_score, roc_auc_score
import sklearn.model_selection as ms
import dask.dataframe as dd
import  dask_ml.model_selection as ds
from lightgbm import LGBMClassifier

import xgboost as xgb
import dask_optuna
import optuna
from dask.distributed import Client



# o = lgbt
# 1= xgb


def objective(trial,x,y):
    xtrain, xval = x
    ytrain, yval = y

    # dtrain = xgb.DMatrix(xtrain, label=ytrain)
    # dvalid = xgb.DMatrix(xval, label=yval)


    fixed_para_xgb = {
        "eval_set":[(xval, yval)],
        # 'eval_set': [ (dvalid, 'valid')],
        "early_stopping_rounds": 5,
        'eval_metric': 'auc',
        'verbose': 25,
        'callbacks': [XGBoostPruningCallback(trial, 'validation_0-auc')],
    }

    param_xgb = {
        # "n_jobs":-1,
        "objective": "binary:logistic",
        "booster": "gbtree",
        # "eta":trial.suggest_float("eta", .01, .5, log=True),
        # "max_depth":trial.suggest_int("max_depth", 2, 8, log=True),
        "n_estimators":trial.suggest_int("n_estimators", 200, 600, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        'use_label_encoder': False
    }




    xboo =xgb.XGBClassifier(**param_xgb)
    xboo.fit(xtrain,ytrain,**fixed_para_xgb)
    preds = xboo.predict_proba(xval)[:,1]



    return roc_auc_score(yval, preds)

def optuna_():
    # train = pd.read_csv("train.csv")
    # test = pd.read_csv("test.csv")

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    X = pd.DataFrame(train.drop(columns=["claim", "id"]))
    xtest = pd.DataFrame(test.drop(columns=["id"]))
    Y = train["claim"]
    folds = 5

    x, xval2, y, yval2 = ms.train_test_split(X, Y, test_size=.1, shuffle=True, random_state=0)
    xtrain, xval, ytrain, yval = ms.train_test_split(x, y, test_size=.2, shuffle=True, random_state=0)
    # xval, xtest, yval, ytest = ms.train_test_split(xval_, yval_, test_size=.5, shuffle=True, random_state=0)



    study = optuna.create_study(direction='maximize', study_name="robin")
    study.optimize(lambda i : objective(i, (xtrain, xval), (ytrain, yval)), n_trials=3)

    print()
    print(study.best_trial.params)
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

    model = xgb.XGBClassifier(**study.best_trial.params)

    for counter, (trn, val) in enumerate(df_split.split(x, y)):
        model.fit(x.iloc[trn, :], y.iloc[trn])
        w = model.predict_proba(xval2)[:, 1]
        score[0, counter] = roc_auc_score(yval2, w)
        predictions[:, counter] = model.predict_proba(xtest.values)[:, 1]
        if counter == folds - 1:
            score = np.reshape(np.dot(1 / np.sum(score), score), (folds, 1))
            print(score)

    final = pd.DataFrame(test["id"])
    final = final.merge(pd.DataFrame(np.dot(predictions, score)), right_index=True, left_index=True)
    final.columns = ["id", "claim"]
    print(final.head(5))

    final.to_csv("sub_v7_xgb.csv", index=False)


def fill_na(X, X_):
    for i in X.columns:
        X_[i] = X_[i].fillna(np.mean(X.loc[X[i].isna() == False, i]))
    return X_




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # main()
    optuna_()
