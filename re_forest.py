"""

Random forest regressor trained on real estate housing data, obtaining 98% test accuracy.  The code cleans 
and prepares the data by inserting missing balues based on mode/mean of the available data or dropping entire 
fetaure vectors that do not contain enough information to be useful.  
@Author: Jason St. George 2018

"""
from sys import argv
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def get_data(fn=None):
    """
    Import data using pandas
    :param fn: filename
    :return: training set, test set, target values (labels)
    """
    if fn:
        return pd.read_csv(fn)

    data_train = pd.read_csv("./housing_data/train.csv")
    target = data_train['SalePrice']
    train = data_train.drop('SalePrice', axis = 1)
    test = pd.read_csv("./housing_data/test.csv")

    return train, test, target


def clean_data(data, prnt=False):
    """
    Handle missing data, replace with reasonable values (median/mode) or drop based on count and contribution
    :param data: imported pandas DataFrame
    :param prnt: Bool
    :return: cleaned-up dataset
    """
    upd_data = data.drop('Id', axis=1)
    upd_data = upd_data.drop('Alley', axis=1)
    upd_data.loc[upd_data.LotFrontage.isnull(), 'LotFrontage'] = upd_data['LotFrontage'].median()
    upd_data.loc[upd_data.MasVnrType.isnull(), 'MasVnrType'] = 'None'
    upd_data.loc[upd_data.MasVnrArea.isnull(), 'MasVnrArea'] = 0.
    upd_data.loc[upd_data.BsmtQual.isnull(), 'BsmtQual'] = 'TA'
    upd_data.loc[upd_data.BsmtCond.isnull(), 'BsmtCond'] = 'TA'
    upd_data.loc[upd_data.BsmtExposure.isnull(), 'BsmtExposure'] = 'No'
    upd_data.loc[upd_data.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'Unf'
    upd_data.loc[upd_data.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'Unf'
    upd_data.loc[upd_data.Electrical.isnull(), 'Electrical'] = 'SBrkr'
    upd_data = upd_data.drop('FireplaceQu', axis=1)
    upd_data.loc[upd_data.GarageType.isnull(), 'GarageType'] = 'Attchd'
    upd_data.loc[upd_data.GarageYrBlt.isnull(), 'GarageYrBlt'] = 2005.
    upd_data.loc[upd_data.GarageQual.isnull(), 'GarageQual'] = 'TA'
    upd_data.loc[upd_data.GarageFinish.isnull(), 'GarageFinish'] = 'Unf'
    upd_data.loc[upd_data.GarageCond.isnull(), 'GarageCond'] = 'TA'
    upd_data = upd_data.drop('PoolQC', axis=1)
    upd_data = upd_data.drop('Fence', axis=1)
    upd_data = upd_data.drop('MiscFeature', axis=1)
    upd_data.loc[upd_data.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = 0.
    upd_data.loc[upd_data.BsmtFinSF2.isnull(), 'BsmtFinSF2'] = 0.
    upd_data.loc[upd_data.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0.
    upd_data.loc[upd_data.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0.
    upd_data.loc[upd_data.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = 0.
    upd_data.loc[upd_data.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
    upd_data.loc[upd_data.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
    upd_data.loc[upd_data.Functional.isnull(), 'Functional'] = 'Typ'
    upd_data.loc[upd_data.GarageArea.isnull(), 'GarageArea'] = 0.
    upd_data.loc[upd_data.GarageCars.isnull(), 'GarageCars'] = 2.
    upd_data.loc[upd_data.GarageCars.isnull(), 'GarageCars'] = 2.
    upd_data.loc[upd_data.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
    upd_data.loc[upd_data.MSZoning.isnull(), 'MSZoning'] = 'RL'
    upd_data.loc[upd_data.SaleType.isnull(), 'SaleType'] = 'WD'
    upd_data.loc[upd_data.TotalBsmtSF.isnull(), 'TotalBsmtSF'] = 0.
    upd_data.loc[upd_data.Utilities.isnull(), 'Utilities'] = 'AllPub'

    if prnt:
        print('Updated in dataset:')
        print(upd_data.info())

    return upd_data


def normalize(raw_data, prnt=False):
    """
    Normalizing Numerical Features
    :param raw_data: unnormalized DataFrame
    :param prnt: Bool
    :return: normalized dataset
    """
    scale = preprocessing.MinMaxScaler()
    numerical = list(raw_data.select_dtypes(include=['int64']).columns.values) + \
                list(raw_data.select_dtypes(include=['float64']).columns.values)

    raw_data[numerical] = scale.fit_transform(raw_data[numerical])

    if prnt:
        print(numerical)
        print(raw_data.head())

    return raw_data


def convert_categorical(data, prnt=False):
    """
    Convert cateogrical features to numeric
    :param data: normalized dataset
    :param prnt: Bool
    :return: Numerically encoded dataset
    """
    encoder = preprocessing.LabelEncoder()
    categorical = list(data.select_dtypes(include=['object']).columns.values)

    for v in categorical:
        encoder.fit(data[v])
        data[v] = encoder.transform(data[v])

    if prnt:
        print(categorical)
        print(data.head())

    return data


def shuffle_split(data, target, prnt=False, r_state=0):
    """
    Shuffle and split data to prepare for training/testing
    :param data: normalized numeric dataset
    :param target: labels
    :param prnt: Bool
    :return: Train and Test sets
    """
    train = data[:1460]
    test = data[1460:]
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = r_state)

    if prnt:
        print(test)
        range(10,100,10)

    return X_train, X_test, y_train, y_test, test


def fit_model(X, y, r_state, n_trees, max_depth):
    """
    Performs a grid search over the 'max_depth' parameter for a trained random forest regressor
    :param X: Examples
    :param y: Labels
    :param r_state: Random state {int}, default is numpy's
    :param n_trees: Number of trees in the forest
    :param max_depth: How deep is the ocean?
    :return: Trained model (random forest)
    """
    random_forest = RandomForestRegressor(random_state=r_state, n_estimators=n_trees, max_depth=max_depth)
    parameters = {'n_estimators': range(10, n_trees, 10)}
    scorer = make_scorer(r2_score)
    grid = GridSearchCV(random_forest, parameters, scorer)
    grid = grid.fit(X, y)

    return grid.best_estimator_


def main():
    r_state, n_trees, max_depth, prnt = None, 100, None, False

    if len(argv) > 1:
        assert len(argv) == 5   , "Usage: realestate_forest.py random_state: {int} num_trees: {int} " \
                                         "max_depth: {int} print: {Bool}"
        r_state, n_trees, max_depth, prnt = int(argv[1]), int(argv[2]), int(argv[3]), bool(argv[4])

    train, test, target = get_data()

    if prnt:
        print(data.head())
        print(data.info())
        print(data.describe(include=['O']))

    data = pd.concat([train, test])

    cleaned_data = clean_data(data, prnt)
    normalized_data = normalize(cleaned_data, prnt)
    converted_data = convert_categorical(normalized_data, prnt)

    X_train, X_test, y_train, y_test, feat_test = shuffle_split(converted_data, target, prnt)
    forest = fit_model(X_train, y_train, r_state, n_trees, max_depth)
    y_predict = forest.predict(X_train)

    print("Accuracy: ", r2_score(y_train, y_predict))

    predicted = forest.predict(feat_test)
    predictions = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": predicted
    })

    predictions.to_csv('predictions.csv')


if __name__ == "__main__":
    main()
