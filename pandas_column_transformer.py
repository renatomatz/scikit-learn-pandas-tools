"""
Implements a modification of the standard sklearn transforrmer aggregator to adapt transformations to pandas DataFrames
"""
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from joblib import Parallel, delayed

from pandas_adapted import PandasAdapted, _pandas_adapted_fto

class PandasColumnTransformer(ColumnTransformer, PandasAdapted):
    """
    Uses as much logic and processes from the original ColumnTransformer but with flexibility of use on Pandas dataframes

    The point of this mod is to enable pipelines to be built using standard sklearn estimators and transformers, applying them 
    to DataFrames and getting a dataframe as output with almost native feel
    """

    def __init__(self,
                 transformers,
                 remainder='drop',
                 sparse_threshold=0.3,
                 n_jobs=None,
                 transformer_weights=None,
                 verbose=False):

        """
        Initiate PCT

        This method uses the same arguments as the original for compatibility but also greates the orig_cols_ variable and makes 
        remainder="passthrough" 
        """

        super().__init__(transformers, 
                         remainder="drop", 
                         sparse_threshold=0.3,
                         n_jobs=n_jobs,
                         transformer_weights=None,
                         verbose=verbose)


    def _fit_transform(self, X, y, func, inplace, fitted=False):
        """
        Private function to fit and/or transform on demand.
        Return value (transformers and/or transformed X data) depends
        on the passed function.
        ``fitted=True`` ensures the fitted transformers are used.

        ADAPTATION: columns are sliced inside func, not outside, assumes func is _pandas_adapted_fto
        """
        # TODO fix the iterator function to propperly take into account fitted transformers 
        # transformers = list(
        #     self._iter(fitted=fitted, replace_strings=True))
        transformers = self.transformers
        try:
            if not inplace:
                X = X.copy() ##--## make a copy of x, which will be mutated on every iteration bellow
            ##--## this call will just mutate X_cp, so there is no need to return the list, just perform
            ##--## the mutations and return X_cp
            Parallel(n_jobs=self.n_jobs)( 
                delayed(func)(
                    transformer=trans,#_check_trans(trans, fitted), ##--##
                    X=X,
                    y=y,
                    column=column,
                    message_clsname='PandasColumnTransformer',
                    message=self._log_message(name, idx, len(self.transformers)))
                for idx, (name, trans, column) in enumerate(transformers, 1))
            return X
        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError("Got 1D array,  DataFrame")
            else:
                raise

    def fit_transform(self, X, y=None, inplace=False):
        """Fit all transformers, transform the data and concatenate results.

        PANDAS ADDITION: RESULTS WILL UPDATE A COPY OF X.
        CHANGES FROM ORIGINAL CODE WILL BE MARKED WITH ##--##

        Parameters
        ----------
        X : array-like or DataFrame of shape [n_samples, n_features]
            Input data, of which specified subsets are used to fit the
            transformers.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        # TODO: this should be `feature_names_in_` when we start having it
        if hasattr(X, "columns"):
            self._feature_names_in = np.asarray(X.columns)
        else:
            self._feature_names_in = None
        ##--## X = _check_X(X) this function normally converts X to an array
        self._validate_transformers()
        self._validate_column_callables(X)
        self._validate_remainder(X)

        result = self._fit_transform(X, y, _pandas_adapted_fto, inplace)

        if result is None:
            self._update_fitted_transformers([])
            # All transformers are None
            return pd.DataFrame(np.zeros((X.shape[0], 0))) ##--##

        # determine if concatenated output will be sparse or not
        ##--## deleted as X's are guaranteed to be dataframes
        ##--## self._update_fitted_transformers(self.transformers)
        #TODO create a method to update fitted transformers
        ##--## self._validate_output(Xs) result validation will be performed inside each call on pandas_adapted_fto

        return result 

    def transform(self, X, y=None):
        """
        For now, this class will not separate fitting and transforming
        TODO separate fitting and transforming
        """
        return self.fit_transform(X, y)

def _check_trans(trans, fitted):
    if fitted or isinstance(trans, PandasAdapted) or trans=="drop":
        return trans
    else:
         clone(trans)

if __name__ == "__main__":
    df_len = 100000 # adjust this to test speed
    df = pd.DataFrame({
        "A": [*range(0, df_len+1, 1)],
        "B": [df_len] + ([0]*(df_len-1) + [np.nan]),
        "C": [*range(0, (df_len+1)*2, 2)]
    })

    from sklearn.impute import SimpleImputer
    pdct = PandasColumnTransformer([
        ("impute_nan", SimpleImputer(strategy="mean"), ["B"]),
        ("drop_A", "drop", ["A"])
    ])

    res = pdct.fit_transform(df)
    print(res.iloc[-1]["B"]) # should be 1 