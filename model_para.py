import pandas as pd
import numpy as np
from optuna.integration import LightGBMPruningCallback
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
pd.set_option('expand_frame_repr', False)
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
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
    def __init__(self, x, y,test, num_iter, fold):
        # self.x, self.xval, self.y, self.yval = ms.train_test_split(X, Y, test_size=.25, shuffle=True, random_state=0)
        self.x = x
        self.y = y
        self.test = test
        self.xval = pd.DataFrame(self.test.drop(columns=["id"]))

        self.folds = fold
        # super().__init__(self.x, self.xval, self.y, self.yval, num_iter)

    def create_base_meta(self):

        with sqlite3.connect("db.sep_tub.DB") as conn:
            para = pd.read_sql("select * from base_para", conn)
            df_split = ms.StratifiedKFold(n_splits=self.folds, shuffle=True)

            meta_val_0 = np.zeros((len(self.xval), self.folds))
            meta_val_1 = np.zeros((len(self.xval), self.folds))
            meta_val_2 = np.zeros((len(self.xval), self.folds))
            meta_val = np.zeros((len(self.xval), len(para.index)))
            train_meta = np.zeros((len(self.x), len(para.index) + 1))

            start = 0
            end = 0
            for counter, (trn, val) in enumerate(df_split.split(self.x, self.y)):
                end += len(val)
                train_meta[start:end, 0] = self.y.iloc[val].values
                print(self.y.iloc[val].values)

                for p in para.itertuples():
                    best_para = json.loads(p.best_para)
                    model = LGBMClassifier(n_jobs=-1, **best_para)
                    model.fit(self.x.iloc[trn, :], self.y.iloc[trn])
                    train_meta[start:end, p.index + 1] = model.predict_proba(self.x.iloc[val, :])[:, 1]
                    if p.index == 0:
                        meta_val_0[:, counter] = model.predict_proba(self.xval)[:, 1]
                    elif p.index== 1:
                        meta_val_1[:, counter] = model.predict_proba(self.xval)[:, 1]
                    elif p.index == 2:
                        meta_val_2 [:, counter] = model.predict_proba(self.xval)[:, 1]

                start +=len(val)

                if counter == self.folds - 1:
                    meta_val[:, 0] = np.mean(meta_val_0, axis=1)
                    meta_val[:, 1] = np.mean(meta_val_1, axis=1)
                    meta_val[:, 2] = np.mean(meta_val_2, axis=1)


        second_model = LogisticRegression(max_iter=10000, solver='saga', n_jobs=-1)
        second_model.fit(train_meta[:, 1:], train_meta[:, 0])
        pred = second_model.predict_proba(meta_val)[:, 1]
        # #
        # print(roc_auc_score(self.yval, pred))

        final = pd.DataFrame(self.test["id"])
        final = final.merge(pd.DataFrame(pred), right_index=True, left_index=True)
        final.columns = ["id", "claim"]
        final.to_csv("sub_v9.csv", index=False)

        print(final.head(5))





























        # weights = []
        #
        # meta_data = np.zeros((len(self.xval2), len(best_par)))
        #
        # df_split = ms.StratifiedKFold(n_splits=self.folds, shuffle=True)
        #
        # for p, m in enumerate(best_par):
        #     score = np.zeros((1, self.folds))
        #
        #     predictions = np.zeros((len(self.xval2), self.folds))
        #     weights.append([best_par[p].m_score])
        #     model = LGBMClassifier(n_jobs=-1, **m.best_para)
        #     for counter, (trn, val) in enumerate(df_split.split(self.x, self.y)):
        #         model.fit(self.x.iloc[trn, :], self.y.iloc[trn])
        #         w = model.predict_proba(self.xval)[:, 1]
        #         score[0, counter] = roc_auc_score(self.yval, w)
        #         predictions[:, counter] = model.predict_proba(self.xval2.values)[:, 1]
        #         if counter == self.folds - 1:
        #             score = np.reshape(np.dot(1 / np.sum(score), score), (self.folds, 1))
        #             print(score)
        #
        #     # if p == len(best_par) - 1:
        #     #     weights = np.array(weights)
        #     #     print(f"auc score{weights}")
        #     #     weights = (weights / np.sum(weights)).reshape(len(best_par), 1)
        #
        #     meta_data[:, p] = (np.dot(predictions, score)).ravel()
        #
        # # pred = np.reshape(np.dot(meta_data, weights), (len(self.xval2), 1))
        # #
        # # print(pred)
        # # model= LogisticRegression()
        # return (pd.DataFrame(meta_data), self.yval2)








        # final = pd.DataFrame(self.test["id"])
        # final = final.merge(pd.DataFrame(pred), right_index=True, left_index=True)
        # final.columns = ["id", "claim"]
        # print(final.head(5))
        #
        #
        # final.to_csv("sub_v8.csv", index=False)
