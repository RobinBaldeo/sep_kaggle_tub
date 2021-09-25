
import pandas as pd
import numpy as np
from model_para import model_best_para, build_base

pd.set_option('expand_frame_repr', False)

def optuna_():

    folds = 10
    num_iter  =100
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    # train =train.loc[0:2000,]
    # test = test.loc[0:2000, ]
    X = pd.DataFrame(train.drop(columns=["claim", "id"]))
    # xtest = pd.DataFrame(test.drop(columns=["id"]))
    Y = train["claim"]


    #
    # x, xval3, y, yval3 = ms.train_test_split(X, Y,test_size=.1, shuffle=True, random_state=0)

    # xtrain, xval, ytrain, yval = ms.train_test_split(x, y, test_size=.2, shuffle=True, random_state=0)
    # # xval, xtest, yval, ytest = ms.train_test_split(xval_, yval_, test_size=.5, shuffle=True, random_state=0)

    mbp = build_base(X, Y,test,   num_iter, folds)
    # base_mo = mbp.dummy()
    sec_mo = mbp.create_base_meta()


def fill_na(X, X_):
    for i in X.columns:
        X_[i] = X_[i].fillna(np.mean(X.loc[X[i].isna() == False, i]))
    return X_




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # main()
    optuna_()
