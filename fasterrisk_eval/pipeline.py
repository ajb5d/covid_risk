import numpy as np
import pandas as pd
import re
from pyspark.sql import functions as F
from fasterrisk.fasterrisk import RiskScoreClassifier

def scorecard_coef_tbl(cardid, master_coef_tbl):
    coeftbl = master_coef_tbl.loc[master_coef_tbl.cardid == cardid, :]

    return coeftbl.sort_values('parmid')


def construct_prediction_df(card_coef_tbl, obsdf):
    """
    Construct a prediction data set for a scorecard.

    :card_coef_tbl: (pandas df): Table of scorecard parameters
    :obsdf: (spark df): Table of observations to use for prediction.
    :return: pandas df of the predictors used in the scorecard.

    """

    ## The table should already be sorted by parmid, but better safe than sorry.
    coeftbl = card_coef_tbl.sort_values('parmid')

    ## drop the multiplier and intercept coefficients; these form part of the
    ## model structure, but they don't correspond to any observed values.
    coeftbl = coeftbl.loc[coeftbl.parmid > 1, :]

    parms = list(coeftbl.parameter)
    print('all parms:\n', parms)

    ## Columns come in two types: those with a simple name are indicator variables
    ## that exist in the data.  Those with a name of the form foo<=X have to be
    ## created.
    pat = re.compile(r'(.+)<=(.+)')
    for p in parms:
        m = pat.match(p)
        ## if no match, nothing to do
        if m is not None:
            col = m.group(1)
            val = float(m.group(2))
            print('p: ', p, '\tcol: ', col, '\tval: ', val)
            #obsdf[p] = np.where(obsdf[col] <= val, 1, 0)
            obsdf = obsdf.withColumn(p, F.when(obsdf[col] <= val, 1).otherwise(0))

    #rslt = obsdf.loc[:, parms]
    parm_escape = ['`' + p + '`' for p in parms]
    print('parm_escape:\n', parm_escape)
    rslt = obsdf.select(parm_escape).toPandas()

    return rslt

def eval_scorecard(cardid, master_coef_tbl, obsdf, yvals):
    """
    :cardid: (int): Index of the card to evaluate
    :master_coef_tbl: (pandas df): Table of coefficients for all scorecards
    :obsdf: (spark df): Table of all patient observations
    :yvals: (numpy 1d): Array of +/- 1 values indicating whether or not the outcome occurred.
    
    """

    card_coef_tbl = scorecard_coef_tbl(cardid, master_coef_tbl)

    xdf = construct_prediction_df(card_coef_tbl, obsdf)

    scorecard = construct_scorecard(card_coef_tbl, xdf)

    return scorecard.compute_logisticLoss(np.asarray(xdf), yvals)

    


def construct_scorecard(card_coef_tbl, X_df):
    """
    Construct a fasterrisk scorecard from the card coefficient table

    :card_coef_tbl: (pandas df): Table of scorecard parameters
    :prediction_df: (pandas df): Data frame of prediction variables needed for the scorecard.
    :return: fasterrisk RiskScoreClassifier object

    """

    coeftbl = card_coef_tbl.sort_values('parmid')

    parms = list(coeftbl['parameter'])
    coefs = np.asarray(coeftbl['coefficient'])

    ## First two entries are special
    multiplier = coefs[0]
    intercept = coefs[1]

    ## The rest of the parameters are in the remainder of the array
    parms = parms[2:]
    coefs = coefs[2:]

    print('multiplier: ', multiplier, '\tintercept: ', intercept)
    print('parms: ', parms)
    print('coefs: ', coefs)

    classifier = RiskScoreClassifier(multiplier, intercept, coefs,
                                     featureNames = parms,
                                     X_train = np.asarray(X_df))

    return classifier


def extract_spark_col(sparkdf, colname):
    "Extract a column of a spark df as a 1-D numpy array"
    return np.asarray(sparkdf.select(colname).toPandas()[colname])


def compute_scorecard_scores(card_coef_tbl, X_df):
    """
    Compute the scores for a scorecard and dataset.

    :card_coef_tbl: (pandas df): table of card coefficients
    :X_df: (pandas df): table of observations
    :return: (numpy array 1d): vector of scores
    
    """

    if any(np.asarray(card_coef_tbl['parameter'])[2:] != np.asarray(list(X_df.columns))):
        err = "column mismatch:\nparameters: {}\ncolumns: {}\n".format(np.asarray(card_coef_tbl['parameter'])[2:],
                                                                       np.asarray(list(X_df.columns)))
        print(err)
        raise RuntimeError(err)
    
    coefs = np.asarray(card_coef_tbl['coefficient'])
    coefs = coefs[2:]   # first two entries are multiplier and intercept, which we don't need.

    rslt = np.asarray(X_df) @ coefs

    return rslt

def theoretical_probs(card_coef_tbl, score):
    """
    Compute the model probability for each score in a score card

    :card_coef_tbl: (pandas df): table of card coefficients
    :score: (numpy 1d): vector of scores
    :return: (numpy 1d): vector of model probabilities
    
    """

    coeftbl = card_coef_tbl.sort_values('parmid')
    coefs = np.asarray(coeftbl['coefficient'])
    mult = coefs[0]
    intercept = coefs[1]

    logit = (intercept + score) / mult

    return 1/(1 + np.exp(-logit))


def calc_bss(prob_tbl):
    """
    Compute Brier Skill Score for a model

    :prob_tbl: (pandas df): table having columns "model_prob", "empirical_prob", and "N",
               giving respectively the probability calculated by the model for a particular score,
               the empirical event probability for that score, and the number of patients with
               that score.
    :return: (float): Brier Skill Score for the model.

    """

    pm = np.asarray(prob_tbl['model_prob'])
    pe = np.asarray(prob_tbl['empirical_prob'])
    ntot = np.asarray(prob_tbl['N'])            # number of cases at each score level

    ne = np.round(ntot * pe)
    nne = ntot - ne

    bs = np.sum(ne*(1 - pm)**2 + nne*(pm**2)) / np.sum(ntot)

    br = np.sum(ne) / np.sum(ntot)

    bsbr = np.sum(ne*(1-br)**2 + nne*(br**2)) / np.sum(ntot)

    bss = 1 - bs/bsbr

    print('bs: ', bs, '\tbsbr: ', bsbr, '\tbss: ', bss)

    return bss

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c41d82e8-a2fb-4507-a808-1d6d5b9133ea"),
    Fr_task_1a=Input(rid="ri.foundry.main.dataset.7192d603-eef1-417d-906c-cc9a99fbdd3a"),
    processed_features_1a_cv9=Input(rid="ri.vector.main.execute.c8041687-96e5-4d2b-a072-0d69243a861f")
)
def fr_1a_selected_scorecard( Fr_task_1a, processed_features_1a_cv9):
    yin = extract_spark_col(processed_features_1a_cv9, 'hosp')
    lloss = [eval_scorecard(i, Fr_task_1a, processed_features_1a_cv9, yin) for i in range(50)]

    ibest = np.argmin(lloss)

    print('best OOS loss: ', lloss[ibest], ' at card ', ibest)
    
    coeftbl = scorecard_coef_tbl(ibest, Fr_task_1a)
    xdf = construct_prediction_df(coeftbl, processed_features_1a_cv9)
    card = construct_scorecard(coeftbl, xdf)

    card.print_model_card()

    return spark.createDataFrame(coeftbl)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.be2d78e5-ecc3-46c7-a73e-0ce5d75921f4"),
    fr_1a_selected_scorecard=Input(rid="ri.foundry.main.dataset.c41d82e8-a2fb-4507-a808-1d6d5b9133ea"),
    processed_features_1a_test=Input(rid="ri.vector.main.execute.c890f9af-d439-4d53-a40f-169d05e45b1c")
)
import sklearn.metrics

def fr_1a_test_set_eval(fr_1a_selected_scorecard, processed_features_1a_test):
    yvals = extract_spark_col(processed_features_1a_test, 'hosp')
    cardid = np.asarray(fr_1a_selected_scorecard['cardid'])[0]         # there is only one card here, so all the cardid values are the same.

    coeftbl = scorecard_coef_tbl(cardid, fr_1a_selected_scorecard)
    xdf = construct_prediction_df(coeftbl, processed_features_1a_test)
    scorecard = construct_scorecard(coeftbl, xdf)

    acc, auc = scorecard.get_acc_and_auc(np.asarray(xdf), yvals)

    scores = compute_scorecard_scores(coeftbl, xdf)
    xdf['score'] = scores
    xdf['outcome'] = np.where(yvals < 0, 0, 1)     # Convert +/- 1 representation to indicator variable
    empirical_prob = xdf[['score','outcome']].groupby('score', as_index=False).agg(['mean','count']).reset_index()
    empirical_prob.columns = ['score','empirical_prob', 'N']
    empirical_prob['model_prob'] = theoretical_probs(coeftbl, np.asarray(empirical_prob['score']))

    print(empirical_prob)

    bss = calc_bss(empirical_prob)

    print("\n**** Test data AUC: ", auc, " ****\n")
    print("\n**** Test data BSS: ", bss, " ****\n")
    
    return spark.createDataFrame(empirical_prob)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.894d028e-20a1-4005-bfea-348adf5db01c"),
    Fr_task_1b_no_fm_vars=Input(rid="ri.foundry.main.dataset.d95f1c2c-8c38-4045-9036-3e9d80487a16"),
    processed_features_1b_dev=Input(rid="ri.vector.main.execute.de52d46d-73f8-4dd7-964e-ac7e654d4c36")
)
def fr_1b_selected_scorecard(Fr_task_1b_no_fm_vars, processed_features_1b_dev):
    yin = extract_spark_col(processed_features_1b_dev, 'deathetc')
    lloss = [eval_scorecard(i, Fr_task_1b_no_fm_vars, processed_features_1b_dev, yin) for i in range(50)]

    ibest = np.argmin(lloss)

    print('best OOS loss: ', lloss[ibest], ' at card ', ibest)
    
    coeftbl = scorecard_coef_tbl(ibest, Fr_task_1b_no_fm_vars)
    xdf = construct_prediction_df(coeftbl, processed_features_1b_dev)
    card = construct_scorecard(coeftbl, xdf)

    return spark.createDataFrame(coeftbl)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.95df0b48-eb91-48cd-9477-3855645c2bc4"),
    fr_1b_selected_scorecard=Input(rid="ri.foundry.main.dataset.894d028e-20a1-4005-bfea-348adf5db01c"),
    processed_features_1b_test=Input(rid="ri.vector.main.execute.ca322e99-581d-4af8-b6a4-314470dda1c1")
)
import sklearn.metrics

def fr_1b_test_set_eval(fr_1b_selected_scorecard, processed_features_1b_test):
    yvals = extract_spark_col(processed_features_1b_test, 'deathetc')
    cardid = np.asarray(fr_1b_selected_scorecard['cardid'])[0]         # there is only one card here, so all the cardid values are the same.

    coeftbl = scorecard_coef_tbl(cardid, fr_1b_selected_scorecard)
    xdf = construct_prediction_df(coeftbl, processed_features_1b_test)
    scorecard = construct_scorecard(coeftbl, xdf)

    acc, auc = scorecard.get_acc_and_auc(np.asarray(xdf), yvals)

    scores = compute_scorecard_scores(coeftbl, xdf)
    xdf['score'] = scores
    xdf['outcome'] = np.where(yvals < 0, 0, 1)     # Convert +/- 1 representation to indicator variable
    empirical_prob = xdf[['score','outcome']].groupby('score', as_index=False).agg(['mean','count']).reset_index()
    empirical_prob.columns = ['score','empirical_prob', 'N']
    empirical_prob['model_prob'] = theoretical_probs(coeftbl, np.asarray(empirical_prob['score']))

    print(empirical_prob)

    bss = calc_bss(empirical_prob)

    print("\n**** Test data AUC: ", auc, " ****\n")
    print("\n**** Test data BSS: ", bss, " ****\n")
    
    return spark.createDataFrame(empirical_prob)

@transform_pandas(
    Output(rid="ri.vector.main.execute.292180b8-8731-4b1f-aec5-460143591a20"),
    Fr_task_1a=Input(rid="ri.foundry.main.dataset.7192d603-eef1-417d-906c-cc9a99fbdd3a"),
    processed_features_1a_test=Input(rid="ri.vector.main.execute.c890f9af-d439-4d53-a40f-169d05e45b1c")
)
def test_scorecard_eval(Fr_task_1a, processed_features_1a_test):
    coeftbl = scorecard_coef_tbl(0, Fr_task_1a)

    print(coeftbl)

    xdf = construct_prediction_df(coeftbl, processed_features_1a_test)

    print(xdf)

    sc = construct_scorecard(coeftbl, xdf)

    sc.print_model_card()
    
    yvals = extract_spark_col(processed_features_1a_test, 'hosp')

    print('dim yvals: ', yvals.shape)

    lloss = eval_scorecard(0, Fr_task_1a, processed_features_1a_test, yvals)

    print('lloss: ', lloss)

    xdf_short = xdf.loc[0:3, :]
    score_short = compute_scorecard_scores(coeftbl, xdf_short)

    pd.set_option('max_columns', None)
    print("xdf_short: \n", xdf_short)
    print("scores: ", score_short)
    

