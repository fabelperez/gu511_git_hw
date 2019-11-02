#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
module: dspipeline.py
Author: zlamberty
Created: 2018-10-19

Description:
    simple data science pipeline functions and a demo which uses these tools to
    model a freely available dataset

Usage:
    import dspipeline
    dspipeline.adult_data_demo()

"""

import argparse
import logging
import logging.config
import os
import shutil
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

from sklearn.exceptions import ConvergenceWarning


# ----------------------------- #
#   Module Constants            #
# ----------------------------- #

# set up a logger to print out log messages
LOGGER = logging.getLogger('dspipeline')
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)-8s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'
        },
        'simple': {
            'format': '%(asctime)s %(levelname)-8s [%(name)s] %(message)s'
        },
        'print': {'format': '%(message)s'}
    },
    'root': {'level': 'DEBUG', 'handlers': ['console']},
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'stream': 'ext://sys.stdout'
        },
        'print': {
            'class': 'logging.StreamHandler',
            'formatter': 'print',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {'print': {'handlers': ['print'], 'propagate': False}}
})


# ----------------------------- #
#   utilities                   #
# ----------------------------- #

def fix_column_names(df):
    """replace dashes with underscores. eventually should handle punctuation"""
    df.columns = [col.lower().replace('-', '_') for col in df.columns]


def my_cv(N=10):
    """factory for creating my prefered cross validation object

    args:
        N (int): the number of folds in our bootstrapping cross validation

    returns
        sklearn.model_selection.StratifiedShuffleSplit: the cv object

    """
    return sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=N,
        test_size=0.2,
        random_state=1337
    )


def cross_validate_scores(pipelines, X, y, cv=None):
    """given an iterable of pipelines, perform cross-validated scoring

    args:
        pipelines (iterable): a list or iterable of sklearn pipelines
        X (nd.array): observed predictors
        y (nd.array): observed targets
        cv (sklearn.model_selection object): a scikit learn cross validation
            object. if `None`, will default to the value defined by `my_cv`

    returns
        pd.DataFrame: dataframe of score information

    raises:
        None
    """
    LOGGER.info(
        'performing cross validated scoring on {} pipelines'.format(
            len(pipelines)
        )
    )
    with warnings.catch_warnings():
        # intentionally surpress some annoying sklearn warning messages
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        cv = cv or my_cv()

        dfscores = pd.DataFrame()

        for p in pipelines:
            fsname = p.steps[0][0]
            mname = p.steps[1][0]

            LOGGER.debug('feature selection method: {}'.format(fsname))
            LOGGER.debug('modelling method: {}'.format(mname))

            score = sklearn.model_selection.cross_validate(
                estimator=p,
                X=X,
                y=y,
                return_train_score=True,
                scoring=('accuracy', 'neg_log_loss'),
                cv=cv,
                n_jobs=-1
            )

            score['fs'] = fsname
            score['m'] = mname
            dfscoresnow = pd.DataFrame(score)

            dfscores = dfscores.append(dfscoresnow, ignore_index=True)

        agg_results = dfscores.groupby(['fs', 'm']).train_neg_log_loss.agg(
            ['mean', 'std']
        )
        LOGGER.debug('cross-validation log loss results:\n{}'.format(agg_results))

        return dfscores


def feature_importance_df(clf, feature_names):
    """create a pandas dataframe of feature importance values

    the scikit learn estimator object `clf` must have a `get_support` method and
    an `estimator_` attribute with a `feature_importances_` attribute (i.e. both
    `clf.get_support()` and `clf.estimator_.feature_importances_` must be
    defined)

    args:
         clf (scikit-learn model object): the scikit learn modelling object
              which has estimated feature importance information
         feature_names (list): a list of feature names for the input columns

    returns:
         pd.DataFrame: dataframe containing feature importance and support info

    raises:
         None

    """
    df_support = pd.DataFrame({
        'feature': feature_names,
        'support': clf.get_support()
    })

    # we only have feature importance for records where `support` is true
    df_support.loc[
        df_support.support, 'importance'
    ] = clf.estimator_.feature_importances_

    df_support = df_support.sort_values(by='importance', ascending=False)

    return df_support


def feature_importance_plot(df_support, height=1200, margin=250):
    """create a plotly barchart showing feature importance values

    args:
         df_support (pd.DataFrame): a dataframe containing support information.
             must have the columns `feature`, `support`, and `importance`
         height (int): height of plotly.graph_objs.Layout object (default: 1200)
         margin (int): margin of plotly.graph_objs.Layout object (default: 250)

    returns:
         plotly.Figure: the plotly figure object which was plotted

    raises:
         None

    """
    # drop the features which weren't chosen, and invert the sort  order (plotly
    # adds bars in this "top to bottom" way for  horizontal bar charts)
    nonzero = df_support[df_support.support].sort_values(by='importance')
    data = [go.Bar(
        x=nonzero.importance,
        y=nonzero.feature,
        orientation='h',
    )]

    # this is
    layout = go.Layout(height=height, margin=go.layout.Margin(l=margin))
    fig = go.Figure(data=data, layout=layout)

    return fig


def get_ccr_df(clf, X, y):
    """calculate cumulative capture rate values given a predictor object and
    observed X and y values

    args:
         clf (scikit-learn model object): the scikit learn modelling object
              which has a predict_proba method
         X (np.array-like): observed predictors
         y (np.array-like): observed targets

    """
    y_pred = clf.predict_proba(X)

    df_ccr = pd.DataFrame({
        'y_obs': y,
        'y_pred': y_pred[:, 1]
    })

    df_ccr = df_ccr.sort_values(by='y_pred', ascending=False)
    ntargets = df_ccr.y_obs.sum()
    df_ccr.loc[:, 'ccr'] = df_ccr.y_obs.cumsum() / ntargets

    # we will also define *perfect* capture and *random* capture
    xarr = np.array(range(df_ccr.shape[0]))
    df_ccr.loc[:, 'x'] = xarr

    yperf = np.ones(xarr.shape)
    yperf[:ntargets] = np.linspace(0, 1, ntargets)
    df_ccr.loc[:, 'perfect'] = yperf

    df_ccr.loc[:, 'random'] = xarr / xarr.max()

    return df_ccr


def make_ccr_plot(df_ccr):
    """create a plotly plot of ccr values

    args:
        df_ccr (pd.DataFrame): a pandas data frame containing ccr values.
            assumes columns named `y_obs`, `y_pred`, `ccr`, `x`, `perfect`, and
            `random`

    returns:
        plotly.graph_objs.Figure: the plotly figure object

    """
    data = [
        # our capture rate
        go.Scatter(
            x=df_ccr.x,
            y=df_ccr.ccr,
            mode='lines',
            line={'width': 2},
            name='our prediction'
        ),
        # random choice
        go.Scatter(
            x=df_ccr.x,
            y=df_ccr.random,
            mode='lines',
            line={
                'dash': 'dash',
                'color': 'black',
                'width': 1,
            },
            name='random'
        ),
        # perfect
        go.Scatter(
            x=df_ccr.x,
            y=df_ccr.perfect,
            mode='lines',
            line={
                'dash': 'dot',
                'color': 'black',
                'width': 1,
            },
            name='perfect'
        )
    ]

    layout = go.Layout(
        title='cumulative captured response',
        xaxis={'title': 'prediction certainty sort index (most certain to least)'},
        yaxis={'title': 'fraction of all true cases obtained'}
    )

    fig = go.Figure(data=data, layout=layout)

    return fig




# ----------------------------- #
#   demo using adult dataset    #
# ----------------------------- #

def load_adult_income_data(test_size=0.2, random_state=1337):
    """load a UCI dataset on adult income

    the adult income dataset is available as part of the [UCI machine learning
    repository](http://archive.ics.uci.edu/ml/index.php).

    we *could* use the requests library to download and parse the column names
    (available [here](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)),
    but instead I've just hard-coded them below.

    also, we *could* use the pre-segregated train and test data sets as our
    train and test, but that would involve some data munging and cleaning that
    is a bit of a mess, and also results in enough data points in our final
    plots that we'd have to change some annoying configuration variables.
    instead, let's pull only the smaller training dataset, and use the
    `scikit-learn` train / test split function to create a test dataset of our
    own.

    args:
        test_size (float): passed to the sklearn train_test_split function to
            determine the fractional size of the test set (default: 0.20)
        random_state (int): passed to the sklearn train_test_split function to
            set the random number generator seed (default: 0.20)

    returns:
        xtrain (np.array): train predictors
        xtest (np.array): test predictors
        ytrain (np.array): train target
        ytest (np.array): test target


    """
    LOGGER.info('downloading adult data set')
    columns = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'target'
    ]

    df = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        names=columns,
        delimiter=', ',
        index_col=False,
        engine='python'
    )
    assert df.shape == (32561, 15)

    # we can use our utility function to fix up some of the column names
    fix_column_names(df)

    # `fnlwgt` and `education_num` should be dropped. Why?
    #     1. `fnlwgt` is a weighting for demographic sampling, and is an
    #        estimate of how many people fall into the given category. we're not
    #        going to use this weighting, so let's get rid of it.
    #     2. `education_num` is a numerical representation of the values in the
    #        `education` column. You could argue that you should keep this
    #        numeric column and drop the `education` column, or convert
    #        `education` into a dummy column and drop `education_num`. we'll do
    #        the latter.
    df = df.drop(['fnlwgt', 'education_num'], axis=1)

    assert df.shape[1] == 13

    # fixing dummy columns
    dummycols = [
        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country'
    ]

    df = pd.get_dummies(
        data=df,
        dummy_na=False,
        columns=dummycols
    )
    assert df.shape[1] == 107

    # fixing column names again, after re-introducing some more noise
    fix_column_names(df)
    assert {
        _ for _ in df.columns if _.startswith('sex')
    } == {'sex_female', 'sex_male'}

    # replace our string-valued target column with numerical 0/1 values:
    df.loc[:, 'target'] = (df.target == '>50K').astype(int)
    assert df.target.unique().tolist() == [0, 1]

    # dropping non-numeric features
    df = df._get_numeric_data()
    assert df.shape[1] == 107

    # `log`-transform of monetary features
    moneycols = [
        'capital_gain',
        'capital_loss'
    ]
    df.loc[:, moneycols] = np.log1p(df[moneycols])
    assert df.capital_gain.max() < 20

    # standardize non-target columns
    nottarget = [col for col in df.columns if col != 'target']
    df.loc[:, nottarget] = sklearn.preprocessing.scale(df[nottarget])
    assert -2 < df.age.min() < 0
    assert 0 < df.age.max() < 4

    # develop a train-test split
    dftrain, dftest = sklearn.model_selection.train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df.target
    )
    assert dftrain.shape[0] == 26048
    assert 0.74 <= (dftest.target == 0).mean() <= 0.76

    xtrain = dftrain[nottarget].values
    ytrain = dftrain['target'].values
    xtest = dftest[nottarget].values
    ytest = dftest['target'].values

    assert xtrain.shape == (26048, 106)
    assert ytrain.shape == (26048,)

    return xtrain, ytrain, xtest, ytest


def adult_data_feature_selectors():
    """create a RFE with random forest estimators and a lasso regression"""
    # RFE with random forests
    LOGGER.info('createing feature selection objects')
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100,
                                                 n_jobs=-1,
                                                 random_state=1337)
    rfe = sklearn.feature_selection.RFE(
        estimator=rf
    )

    # lasso
    lr = sklearn.linear_model.LogisticRegression(
        C=.1,
        penalty='l1',
        solver='saga',
        n_jobs=-1,
        random_state=1337,
        max_iter=250
    )
    lasso = sklearn.feature_selection.SelectFromModel(estimator=lr)

    return {'rfe': rfe, 'lasso': lasso}


def adult_data_modellers():
    """a handful of modelers"""
    LOGGER.info('createing data modeller objects')
    cv = my_cv()
    # random forest
    mrf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=1337,
    )
    # logistic regression
    mLogRegCv = sklearn.linear_model.LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 7),
        cv=cv,
        scoring='neg_log_loss',
        n_jobs=-1,
        max_iter=500,
        random_state=1337,
        verbose=0
    )
    # gradient boosting
    mGB = sklearn.ensemble.GradientBoostingClassifier(
        random_state=1337,
    )

    return {
        'random_forest': mrf,
        'logistic': mLogRegCv,
        'gradient_boosting': mGB
    }


def adult_data_demo():
    xtrain, ytrain, xtest, ytest = load_adult_income_data(
        test_size=0.2, random_state=1337
    )

    # feature selection options
    fs_dict = adult_data_feature_selectors()

    # modelers
    model_dict = adult_data_modellers()

    # create a pipeline from the above
    cachedir = tempfile.mkdtemp()
    memory = joblib.Memory(cachedir, verbose=0)
    pipelines = [
        sklearn.pipeline.Pipeline(
            steps=[
                (fsname, fs),
                (modelname, model)
            ],
            memory=memory,
        )
        for (fsname, fs) in fs_dict.items()
        for (modelname, model) in model_dict.items()
    ]

    # cross validation
    df_scores = cross_validate_scores(pipelines, xtrain, ytrain)

    # select the pair with the best overall negative log loss
    fs, m = df_scores.groupby(['fs', 'm']).mean().test_neg_log_loss.idxmax()
    p_best = [
        p for p in pipelines
        if fs in p.named_steps
        and m in p.named_steps
    ][0]

    # re-fit this model to the *entire* train data (it has only ever been fitted
    # to bootstrapped sub-samples)
    p_best.fit(xtrain, ytrain)

    # get ccr values on test data
    df_ccr = get_ccr_df(p_best, xtest, ytest)
    ccr_fig = make_ccr_plot(df_ccr)

    # clean up our pipeline memory
    shutil.rmtree(cachedir)

    return p_best, df_scores, df_ccr, ccr_fig

