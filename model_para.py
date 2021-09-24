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

class model_best_para:
    def __init__(self, xtrain, xval, ytrain, yval, num_iter):
        self.xtrain = xtrain
        self.xval = xval
        self.yval = yval
        self.ytrain = ytrain
        self.num_iter = num_iter

    def objective(self, trial, x, y):
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
            "objective": "binary",
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
            study.optimize(lambda i: self.objective(i, (self.xtrain, self.xval), (self.ytrain, self.yval, n)),
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
        # X, self.weight_x, Y, self.weight_y = ms.train_test_split(X_, Y_, test_size=.1, shuffle=True, random_state=0)
        # self.x, self.xval, self.y, self.yval = ms.train_test_split(X_, Y_, test_size=.20, shuffle=True, random_state=0)

        self.x = X_
        self.y = Y_
        self.test = test
        self.xval = pd.DataFrame(self.test.drop(columns=["id"]))

        self.folds = fold
        self.df_split = ms.StratifiedKFold(n_splits=self.folds, shuffle=True)
        # super().__init__(self.x, self.xval, self.y, self.yval, num_iter)

    def create_base_meta(self):

        with sqlite3.connect("db.sep_tub.DB") as conn:
            # para = pd.read_sql("select model_, best_para, m_score from base_para where model_ in ('rf', 'gbdt')", conn)
            para = pd.read_sql("select model_, best_para, m_score from base_para", conn)
            print(para)
            print(len(para.index))
            # df_split = ms.StratifiedKFold(n_splits=self.folds, shuffle=True)

            meta_val_0 = np.zeros((len(self.xval), self.folds))
            meta_val_1 = np.zeros((len(self.xval), self.folds))
            meta_val_2 = np.zeros((len(self.xval), self.folds))
            meta_val = np.zeros((len(self.xval), len(para.index)))
            train_meta = np.zeros((len(self.x), len(para.index) + 1))

            score_0 = np.zeros((1, self.folds))
            score_1 = np.zeros((1, self.folds))

            start = 0
            end = 0
            for counter, (trn, val) in enumerate(self.df_split.split(self.x, self.y)):
                end += len(val)
                train_meta[start:end, 0] = self.y.iloc[val].values
                print(self.y.iloc[val].values)

                for p in para.itertuples():
                    best_para = json.loads(p.best_para)
                    model = LGBMClassifier(n_jobs=-1, **best_para)
                    model.fit(self.x.iloc[trn, :], self.y.iloc[trn])
                    train_meta[start:end, p.Index + 1] = model.predict_proba(self.x.iloc[val, :])[:, 1]
                    if p.Index == 0:
                        meta_val_0[:, counter] = model.predict_proba(self.xval)[:, 1]
                        # score_0[0, counter] = roc_auc_score(self.weight_y, model.predict_proba(self.weight_x)[:, 1])
                    elif p.Index== 1:
                        meta_val_1[:, counter] = model.predict_proba(self.xval)[:, 1]
                        # score_1[0, counter] = roc_auc_score(self.weight_y, model.predict_proba(self.weight_x)[:, 1])
                    elif p.Index ==2:
                        meta_val_2[:, counter] = model.predict_proba(self.xval)[:, 1]

                start +=len(val)

                if counter == self.folds - 1:
                    # meta_val[:, 0] = (np.dot(meta_val_0, (score_0 / np.sum(score_0)).reshape(self.folds, 1))).ravel()
                    # meta_val[:, 1] = (np.dot(meta_val_1, (score_1 / np.sum(score_1)).reshape(self.folds, 1))).ravel()
                    # meta_val[:, 2] = (np.dot(meta_val_1, (score_1 / np.sum(score_1)).reshape(self.folds, 2))).ravel()
                    meta_val[:, 0] = np.mean(meta_val_0, axis=1)
                    meta_val[:, 1] = np.mean(meta_val_1, axis=1)
                    meta_val[:, 2] = np.mean(meta_val_2, axis=1)

            second_model = LogisticRegression(max_iter=10000, solver='saga', n_jobs=-1, penalty='none')

            second_model.fit(train_meta[:, 1:], train_meta[:, 0])
            pred = second_model.predict_proba(meta_val)[:, 1]
            # #
            # print(f"coco {roc_auc_score(self.yval, pred)}")
            # print(f"gbdt {roc_auc_score(self.yval, meta_val[:, 1])}")
            #
            # train_meta = pd.DataFrame(train_meta)
            # train_meta.columns = ["claim", "gbdt", "rf", "goss"]
            #
            # test_meta = pd.DataFrame(meta_val)
            # test_meta.columns = ["gbdt", "rf", "goss"]
            # test_meta = test_meta.merge(self.yval.reset_index(), right_index=True, left_index=True)
            # test_meta = test_meta.drop(columns="index")


            # train_meta.to_sql("train_meta", conn, if_exists="replace")
            # test_meta.to_sql("test_meta", conn, if_exists="replace")

            final = pd.DataFrame(self.test["id"])
            final = final.merge(pd.DataFrame(pred), right_index=True, left_index=True)
            final.columns = ["id", "claim"]
            final.to_csv("sub_v13.csv", index=False)

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













        # final = pd.DataFrame(self.test["id"])
        # final = final.merge(pd.DataFrame(pred), right_index=True, left_index=True)
        # final.columns = ["id", "claim"]
        # final.to_csv("sub_v10.csv", index=False)

        # print(final.head(5))






















