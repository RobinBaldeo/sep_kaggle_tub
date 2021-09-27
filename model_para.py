import pandas as pd
import numpy as np
from optuna.integration import LightGBMPruningCallback
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
pd.set_option('expand_frame_repr', False)
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import optuna
from collections import namedtuple
import  sqlite3
import json
from sklearn.linear_model import SGDClassifier

class model_best_para:
    def __init__(self, xtrain, xval, ytrain, yval, num_iter):
        self.xtrain = xtrain
        self.xval = xval
        self.yval = yval
        self.ytrain = ytrain
        self.num_iter = num_iter

    def rf_gosss_gbdt(self, trial, x, y):
        xtrain, xval = x
        ytrain, yval, j = y

        fixed_para = {
            "early_stopping_rounds": 5,
            'eval_metric': 'auc',
            'eval_set': [(xval, yval)],
            'verbose': 25,
            'callbacks': [LightGBMPruningCallback(trial, 'auc')],
        }

        param = {
            "objective": trial.suggest_categorical("objective", ["binary","cross_entropy"]),
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": j,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
            # "max_depth":trial.suggest_int("max_depth", 1, 8, log=True),
            # "learning_rate": trial.suggest_float("learning_rate", 0.01, 1., log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 5., log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 5., log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        if param["boosting_type"] in ['rf', 'gbdt']:
            param["feature_fraction"] = trial.suggest_float("feature_fraction", 0.4, 1.0)
            param["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)
            param["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)

        gbm = LGBMClassifier(**param)
        gbm.fit(xtrain, ytrain, **fixed_para)
        preds = gbm.predict_proba(xval)[:, 1]

        return roc_auc_score(yval, preds)


    def para(self):
        para = namedtuple("para", "model_ best_para m_score")
        lst = []
        for n in ['goss', 'gbdt', 'rf']:
            study = optuna.create_study(direction='maximize', study_name="robin")
            study.optimize(lambda i: self.rf_gosss_gbdt(i, (self.xtrain, self.xval), (self.ytrain, self.yval, n)),
                           n_trials=self.num_iter)

            lst.append(para(model_=n, best_para=json.dumps(study.best_trial.params),
                            m_score=([j for i, j in study.best_trial.intermediate_values.items()])[-1]))
            # print(f"best trial {n}: {([j for i, j in study.best_trial.intermediate_values.items()])[-1]}")
            # print(f"Trial #{study.best_trial.number} best trial: {study.best_trial.params}")

        # adding parameters to sql lite
        with sqlite3.connect("db.sep_tub.DB") as conn:
            tbl = pd.DataFrame(lst)
            tbl.to_sql("base_para", conn, if_exists="replace")


        return lst


class build_base(model_best_para):
    def __init__(self, X_, Y_,test, num_iter, fold):
        self.x, self.weight_x, self.y, self.weight_y = ms.train_test_split(X_, Y_, test_size=.05, shuffle=True, random_state=0)

        self.test = test
        self.xval = pd.DataFrame(self.test.drop(columns=["id"]))

        self.folds = fold
        self.df_split = ms.StratifiedKFold(n_splits=self.folds, shuffle=True)
        # super().__init__(self.x, self.weight_x, self.y, self.weight_y, num_iter)

    # def dummy(self):
    #     super().para()
    #     pass


    def create_base_meta(self):
        dt = namedtuple("dt", "model_ best_para")
        para = []

        para.append(dt(model_="goss", best_para={"boosting_type":"goss", "objective": "cross_entropy", "n_estimators": 878, "lambda_l1": 0.02119367084330647, "lambda_l2": 9.259284311814404e-05, "num_leaves": 85, "min_child_samples": 42, "verbose" :-2}))
        para.append(dt(model_="rf", best_para={"boosting_type":"rf","n_estimators": 327, "lambda_l1": 0.00012043760866269098, "lambda_l2": 6.649019338833096e-06, "num_leaves": 246, "min_child_samples": 99, "feature_fraction": 0.7734184326473208, "bagging_fraction": 0.999835036473764, "bagging_freq": 3, "verbose" :-2}))
        para.append(dt(model_="gbdt", best_para={"boosting_type":"gbdt","n_estimators": 499, "lambda_l1": 1.0450194511913434e-06, "lambda_l2": 2.2690854683431152e-07, "num_leaves": 110, "min_child_samples": 14, "feature_fraction": 0.7468626653258925, "bagging_fraction": 0.9944777742119832, "bagging_freq": 4, "verbose" :-2}))

        para = pd.DataFrame(para)


        meta_val = np.zeros((len(self.xval.index) * len(para.index), self.folds))
        meta_val_ave = np.zeros((len(self.xval.index), len(para.index)))
        weight = np.zeros((len(para.index), self.folds))
        val_len = len(self.xval.index)


        train_meta = np.zeros((len(self.x.index), len(para.index) + 1))

        start = 0
        end = 0
        for counter, (trn, val) in enumerate(self.df_split.split(self.x, self.y)):
            end += len(val)
            train_meta[start:end, 0] = self.y.iloc[val].values


            for p in para.itertuples():
                model = LGBMClassifier(n_jobs=-1, **p.best_para)
                model.fit(self.x.iloc[trn, :], self.y.iloc[trn])
                train_meta[start:end, p.Index + 1] = model.predict_proba(self.x.iloc[val, :])[:, 1]
                weight[p.Index, counter] = roc_auc_score(self.weight_y, model.predict_proba(self.weight_x)[:, 1])
                meta_val[val_len * p.Index:val_len * (p.Index + 1), counter] = model.predict_proba(self.xval)[:, 1]
            start +=len(val)

            if counter == self.folds - 1:

                for r in range(0,3):
                    mv = meta_val[val_len * r:val_len * (r + 1),]
                    sc = weight[r,]
                    # print(np.dot(mv, (sc / np.sum(sc)).reshape(self.folds, 1)))
                    meta_val_ave[:, r] = np.dot(mv, (sc / np.sum(sc)))
                    print(f"{para.loc[r, 'model_']} with score {sc}")


        second_model2 = SGDClassifier(max_iter=10000, loss='log')
        #
        second_model2.fit(train_meta[:, 1:], train_meta[:, 0])
        pred = second_model2.predict_proba(meta_val_ave)[:, 1]
        # print(pred)


        final = pd.DataFrame(self.test["id"])
        final = final.merge(pd.DataFrame(pred), right_index=True, left_index=True)
        final.columns = ["id", "claim"]
        final.to_csv("sub_v19.csv", index=False)

        print(final.head(5))










    def op_log_reg(self, trial):
        with sqlite3.connect("db.sep_tub.DB") as conn:
            train_meta = pd.read_sql("select claim, gbdt, rf from train_meta", conn)
        ytrain_m = train_meta["claim"]
        xtrain_m = train_meta.drop(columns="claim")
        X, xval, Y, yval = ms.train_test_split(xtrain_m, ytrain_m, test_size=.1, shuffle=True, random_state=0)

        param = {
            "penalty":trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
            "solver":"saga",
            "max_iter":10000
        }

        if param["penalty"] in ['l1', 'l2', 'elasticnet']:
            param["C"] = trial.suggest_float("C", .01, 20, log=True)

        if param["penalty"] in ['elasticnet']:
            param["l1_ratio"] = trial.suggest_float("l1_ratio", .0001, 1., log=True)

        gbm = LogisticRegression(**param)
        gbm.fit(X, Y)
        preds = gbm.predict_proba(xval)[:, 1]
        return roc_auc_score(yval.values, preds)

    def tune_meta_model(self):
        study = optuna.create_study(direction='maximize', study_name="robin")
        study.optimize(self.op_log_reg,n_trials=20)
        print(f"Trial #{study.best_trial.number} best trial: {study.best_trial.params}")



    def create_second_model(self):
        self.create_base_meta()
        with sqlite3.connect("db.sep_tub.DB") as conn:
            train_meta = pd.read_sql("select claim, gbdt, rf, goss from train_meta", conn)
            test_meta = pd.read_sql("select claim, gbdt, rf, goss from test_meta", conn)

        ytrain_m = train_meta["claim"]
        xtrain_m = train_meta.drop(columns="claim")

        ytest_m = test_meta["claim"]
        xtest_m = test_meta.drop(columns="claim")



        second_pred = np.zeros((len(ytest_m.index),self.folds))

        meta_model = LogisticRegression(max_iter=10000, solver='saga', n_jobs=-1, penalty='none')
        for counter, (trn, val) in enumerate(self.df_split.split( xtrain_m, ytrain_m)):
            meta_model.fit(xtrain_m.iloc[trn,:], ytrain_m.iloc[trn])
            second_pred[:,counter] = meta_model.predict_proba(xtest_m.values)[:,1]

        second_pred = np.mean(second_pred, axis=1)
        print(roc_auc_score(ytest_m.values, second_pred))





























