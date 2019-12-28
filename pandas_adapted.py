"""
This is a base class for all transformers adapted to take in and output Pandas dataframes

This will mostly be used for better indentification of these custom transformers and to create methods to be used in place
of originals when overwriting original sklearn functions

All methods, function and classes overwriting originals will have their original function descriptions as well as a note
named ADAPTATIONS summarizing how the specific object was changed

If only small parts of the object are changed, changes or areas on which changes occur will be symbolized with ##--##
"""

import pandas as pd

from sklearn.utils import _print_elapsed_time

class PandasAdapted:
    """
    PandasAdapted transformers must take in a pandas dataframe as an argument for their .fit and .transform methods
    and return a modified version of that dataframe

    We assume that any modifications dependent on columns will have said columns set upon initialization

    PandasAdapted transformers WILL NOT RETURN SLICES OF THE ORIGINAL DATAFRAME UNLESS THAT IS HOW IT IS SUPPOSED TO BE 
    USED AS A DATAFRAME FROM THE TRANSFORMATION ONWARDS

    All PandasAdapted transformers should have an inplace parameter for their fit_transform/transform methods
    """
    pass

def _pandas_adapted_fto(transformer,
                       X,
                       y,
                       column,
                       message_clsname='',
                       message=None,
                       **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, 
    
    ##--## the result will be multiplied by ``weight`` ##--## not anymore, as X is a dataframe

    ADAPTATION: this performs the transformation on the segment of X needed and mutates X to include them
    """
    with _print_elapsed_time(message_clsname, message):
        if transformer == "drop":
            X.drop(column, axis=1, inplace=True)
            return X

        elif isinstance(transformer, PandasAdapted): ##--## if the transformer already adapts for a dataframe, just update X's columns and return
            transformer.fit_transform(X, y, inplace=True, **fit_params)

        else: ##--## this now prepares result to be a dataframe
            if hasattr(transformer, 'fit_transform'):
                res = transformer.fit_transform(X[column], y, **fit_params)
            else:
                res = transformer.fit(X[column], y, **fit_params).transform(X)

    ##--## from this point, on, I update the columns of X and return that
    try:
        X[column] = pd.DataFrame(res, columns=column)
        ##--## no need to return, as this just mutates X
    except ValueError as e:
        raise ValueError("Output of {} should be a dataframe".format(message))