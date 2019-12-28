"""Transformers to be used in pipeline
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer #, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder

class DropNA(BaseEstimator, TransformerMixin):
    """Imputer to drop rows with missing data
    """
    na_rows_: list
    cols: list

    def __init__(self, cols=None):
        self.na_rows_ = []
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]

    def fit(self, data, y=0):
        for i, row in (data[self.cols] if self.cols is not None else data).iterrows():
            if row.isna().agg(sum) > 0:
                self.na_rows_.append(i)

    def transform(self, data, y=0):
        return data.copy().drop(self.na_rows_, axis=0)


class Mutate(BaseEstimator, TransformerMixin):
    """Create a new column by applying a function over other rows of other columns

    f: a function that takes in a subset of a dataframe and returns a series to be used as the new column s.t.
    cols: columns to be processed
    new_colname: name of the new column, if none is specified, a generic name will be given based on how many no names have been created on the past
    new_col_: created uppon fitting
    one_hot: whether to return one-hot-encoded columns of the new mutated column instead of just the mutation
    """

    f: object
    new_colname: str
    new_col_: object #pd.Series
    one_hot: bool

    id = 0

    def __init__(self, f, new_colname=None, one_hot=False):
        self.f = f
        
        if new_colname is not None:
            self.new_colname = new_colname
        else:
            # self.new_colname = "new_col_{}".format(id := id+1) 
            # please get this back when python 3.8 is more used, hail the walrus
            self.new_colname = "new_col_{}".format(Mutate.id) 
            Mutate.id += 1

    def fit(self, data, y=0):
        self.new_col_ = self.f(data)

    def transform(self, data, y=0):
        # return data[new_colname] := self.new_col_
        # please get this back when python 3.8 is more used, hail the walrus
        data[self.new_colname] = self.new_col_

        if not self.one_hot:
            return data
        else:
            return OneHotEncoder().fit_transform(data)


class PivotCols(BaseEstimator, TransformerMixin):
    """Spread columns to make columns out of categorical values

    columns: list of columns to spread
    values: values significant to these categories
    imputer: Transformer used to impute possible missing values after pivot
    """

    columns: list
    values: list
    imputer: object #sklearn.Transformer

    def __init__(self, columns, values, impute_with=None):
        """Initialize object

        impute_with: how to impute missing values, can be one of
        - None, in which case, missing values will remain untouched
        - a constant to imput missing values with (imputer will be created)
        - a SimpleImputer strategy ("mean", "median", "most_frequent")
        - a pre-created imputer
        """
        self.columns = columns
        self.values = values

        if impute_with is None:
            self.imputer = None
        elif isinstance(impute_with, SimpleImputer):
                # isinstance(impute_with, KNNImputer) or \
                # isinstance(impute_with, IterativeImputer): still experimental
            self.imputer = impute_with
        elif impute_with in ["mean", "median", "most_frequent"]:
            self.imputer = SimpleImputer(strategy=impute_with)
        else:
            self.imputer = SimpleImputer(strategy="constant", fill_value=impute_with)

    def fit(self, data, y=0):
        pass

    def transform(self, data, y=0):
        if self.imputer is not None:
            return self.imputer.fit_transform(
                pd.pivot_table(data, columns=self.columns, values=self.values))
        else:
            return pd.pivot_table(data, columns=self.columns, values=self.values)
