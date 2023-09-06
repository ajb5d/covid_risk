

@transform_pandas(
    Output(rid="ri.vector.main.execute.d5927a3f-88a8-47e7-bce3-630f8461c12b"),
    model_1a_score_test_preds=Input(rid="ri.foundry.main.dataset.e58c09a6-59a2-4751-9d1e-5d3460badb81")
)
def eval(model_1a_score_test_preds):
    test_preds = model_1a_score_test_preds
    from sklearn.metrics import roc_auc_score

    ytrue = test_preds['hosp']
    ypred = test_preds['pred']

    print(roc_auc_score(ytrue, ypred))

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.89fd1198-d34e-416b-9f53-812829b3c57b"),
    fr_features_1a=Input(rid="ri.vector.main.execute.3c9085d3-5893-46ed-88b9-581222e25ef6")
)
import time
import numpy as np
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier

def fr_discretized_test(fr_features_1a):
    feat = np.asarray(fr_features_1a)
    xtrain = feat[:, 0:-1]       # outcome is in the last column ('hosp')
    ytrain = feat[:, -1]

    ## model training parameters -- I just lifted these from the example
    sparsity = 5
    parent_size = 10

    ## Fit model
    model = RiskScoreOptimizer(X=xtrain, y=ytrain, k=sparsity, parent_size=parent_size)

    t1 = time.time()
    model.optimize()
    t2 = time.time()

    print('Optimizaiton time: ', t2-t1)
    
    ## Create a classifier from the first model (evidently a bunch of models are fit?)
    multipliers, beta0s, betas = model.get_models()
    print("{} risk score models fit from the sparse diverse pool".format(len(multipliers)))
    m = multipliers[0]
    intercept = beta0s[0]
    coefs = betas[0]
    classifier = RiskScoreClassifier(m, intercept, coefs, X_train = xtrain)

    ## We would want to supply the classifier with some out of sample data here.  I think
    ## that if we do, then the model will update its empirical risk estimates with the
    ## out of sample risk, but it's unclear from the examples.

    featurenames = list(fr_features_1a.columns[0:-1])
    classifier.reset_featureNames(featurenames)
    classifier.print_model_card()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7192d603-eef1-417d-906c-cc9a99fbdd3a"),
    processed_feat_1a=Input(rid="ri.vector.main.execute.e16624d9-44cd-41a8-b257-e3a3390ef3e6")
)
import time
import numpy as np
import pandas as pd
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df

def fr_task_1a(processed_feat_1a):

    (xtrain, ytrain, featurenames) = fr_prep_data(processed_feat_1a)    

    ## model training parameters -- I just lifted these from the example
    sparsity = 10
    parent_size = 10

    ## Fit model
    model = RiskScoreOptimizer(X=xtrain, y=ytrain, k=sparsity, parent_size=parent_size)

    t1 = time.time()
    model.optimize()
    t2 = time.time()

    print('Optimization time: ', t2-t1)
    
    ## Create a classifier from the first model (evidently a bunch of models are fit?)
    multipliers, beta0s, betas = model.get_models()
    print("{} risk score models fit from the sparse diverse pool".format(len(multipliers)))
    m = multipliers[0]
    intercept = beta0s[0]
    coefs = betas[0]
    print("m= {}\tintercept= {}\ncoefs= {}\n".format(m, intercept, coefs))
    classifier = RiskScoreClassifier(m, intercept, coefs, X_train = xtrain)

    ## We would want to supply the classifier with some out of sample data here.  I think
    ## that if we do, then the model will update its empirical risk estimates with the
    ## out of sample risk, but it's unclear from the examples.

    classifier.reset_featureNames(featurenames)
    classifier.print_model_card()

    rslt_tbl = tabulate_scorecards(multipliers, beta0s, betas, featurenames)

    return(spark.createDataFrame(rslt_tbl))

def fr_prep_data(indata):
    ytrain = np.asarray(indata.iloc[:, -1])       # outcome in the last column

    ## discretize the training variables
    features_discretized = convert_continuous_df_to_binary_df(indata.iloc[:, 0:-1])
    print('feature columns:\n')
    print(list(features_discretized.columns))
    print('len: ', len(list(features_discretized.columns)))

    xtrain = np.asarray(features_discretized)
    print(xtrain)
    print(xtrain.shape)

    featurenames = list(features_discretized.columns)

    return(xtrain, ytrain, featurenames)

def tabulate_scorecards(multipliers, beta0s, betas, featurenames):
    def scorecard_table(i):
        coefs = [multipliers[i], beta0s[i]] + list(betas[i])
        parms = ['(multiplier)', '(intercept)'] + featurenames
        cardidx = [i] * len(coefs)
        parmidx = range(len(coefs))

        tbl = pd.DataFrame({'cardid':cardidx, 'parmid':parmidx, 'parameter':parms, 'coefficient':coefs})

        ## Drop any beta coefficients that are zero (which should be most of them).
        ## Always keep multiplier and intercept
        eps = 1e-6              # Coefficients should be integers, but this saves us headaches if the zeros aren't exact.
        foo = list(abs(np.asarray(betas[i])))        
        keep = np.asarray([True, True] + list(abs(np.asarray(betas[i])) > eps))

        return tbl.loc[keep, :]

    return(pd.concat(map(scorecard_table, range(len(multipliers)))))

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bc5f0217-4c58-4a7f-8155-fd438be6af7e"),
    processed_features_1b=Input(rid="ri.foundry.main.dataset.2ea7125a-95d8-4123-868c-ec3136efee48")
)
import time
import numpy as np
import pandas as pd
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df

def fr_task_1b_all_vars(processed_features_1b):
    
    (xtrain, ytrain, featurenames) = fr_prep_data(processed_features_1b)    

    ## model training parameters -- I just lifted these from the example
    sparsity = 15
    parent_size = 15

    ## Fit model
    model = RiskScoreOptimizer(X=xtrain, y=ytrain, k=sparsity, parent_size=parent_size)

    t1 = time.time()
    model.optimize()
    t2 = time.time()

    print('Optimization time: ', t2-t1)
    
    ## Create a classifier from the first model (evidently a bunch of models are fit?)
    multipliers, beta0s, betas = model.get_models()
    print("{} risk score models fit from the sparse diverse pool".format(len(multipliers)))
    m = multipliers[0]
    intercept = beta0s[0]
    coefs = betas[0]
    print("m= {}\tintercept= {}\ncoefs= {}\n".format(m, intercept, coefs))
    classifier = RiskScoreClassifier(m, intercept, coefs, X_train = xtrain)

    ## We would want to supply the classifier with some out of sample data here.  I think
    ## that if we do, then the model will update its empirical risk estimates with the
    ## out of sample risk, but it's unclear from the examples.

    classifier.reset_featureNames(featurenames)
    classifier.print_model_card(quantile_len = 10)

    rslt_tbl = tabulate_scorecards(multipliers, beta0s, betas, featurenames)

    return(spark.createDataFrame(rslt_tbl))

def fr_prep_data(indata):
    indata = indata.dropna('any').toPandas()
    ytrain = np.asarray(indata.iloc[:, -1])       # outcome in the last column

    ## discretize the training variables
    features_discretized = convert_continuous_df_to_binary_df(indata.iloc[:, 0:-1])
    print('feature columns:\n')
    print(list(features_discretized.columns))
    print('len: ', len(list(features_discretized.columns)))

    xtrain = np.asarray(features_discretized)
    print(xtrain)
    print(xtrain.shape)

    featurenames = list(features_discretized.columns)

    return(xtrain, ytrain, featurenames)

def tabulate_scorecards(multipliers, beta0s, betas, featurenames):
    def scorecard_table(i):
        coefs = [multipliers[i], beta0s[i]] + list(betas[i])
        parms = ['(multiplier)', '(intercept)'] + featurenames
        cardidx = [i] * len(coefs)
        parmidx = range(len(coefs))

        tbl = pd.DataFrame({'cardid':cardidx, 'parmid':parmidx, 'parameter':parms, 'coefficient':coefs})

        ## Drop any beta coefficients that are zero (which should be most of them).
        ## Always keep multiplier and intercept
        eps = 1e-6              # Coefficients should be integers, but this saves us headaches if the zeros aren't exact.
        foo = list(abs(np.asarray(betas[i])))        
        keep = np.asarray([True, True] + list(abs(np.asarray(betas[i])) > eps))

        return tbl.loc[keep, :]

    return(pd.concat(map(scorecard_table, range(len(multipliers)))))

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d95f1c2c-8c38-4045-9036-3e9d80487a16"),
    processed_features_1b=Input(rid="ri.foundry.main.dataset.2ea7125a-95d8-4123-868c-ec3136efee48")
)
import time
import numpy as np
import pandas as pd
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier
from fasterrisk.binarization_util import convert_continuous_df_to_binary_df

def fr_task_1b_no_fm_vars(processed_features_1b):
    usefeatures = processed_features_1b.select(
        'age', 'SexFemale',
        'tuberculosis', 'liver_mld', 'liver_mod', 'thalassemia',
	'rheum', 'dementia', 'chf', 'substance', 'downsynd', 'kidney',
	'maligcanc', 'diabcx', 'diabun', 'cerebvasc', 'periphvasc', 'pregnancy',
	'hf', 'paralysis', 'psychosis', 'obesity', 'coronary', 'csteroid',
	'depression', 'metacanc', 'hiv', 'chroniclung', 'peptic_ulcer', 'sicklecell',
	'mi', 'cardiomyopathy', 'htn', 'immunodef', 'pulm_emb', 'tobacco', 'transplant',
        'isvaxed',
        'antibody_pos', 'antibody_neg',
        'poverty_pct', 'race_white_pct', 'edu_no_hs_pct', 'health_insurance_pct', 'md_per_thous_pop',
        'treat_steroid', 'remdisivir',
        'deathetc'
    )
    (xtrain, ytrain, featurenames) = fr_prep_data(usefeatures)    

    ## model training parameters -- I just lifted these from the example
    sparsity = 20
    parent_size = 10

    ## Fit model
    model = RiskScoreOptimizer(X=xtrain, y=ytrain, k=sparsity, parent_size=parent_size)

    t1 = time.time()
    model.optimize()
    t2 = time.time()

    print('Optimization time: ', t2-t1)
    
    ## Create a classifier from the first model (evidently a bunch of models are fit?)
    multipliers, beta0s, betas = model.get_models()
    print("{} risk score models fit from the sparse diverse pool".format(len(multipliers)))
    m = multipliers[0]
    intercept = beta0s[0]
    coefs = betas[0]
    print("m= {}\tintercept= {}\ncoefs= {}\n".format(m, intercept, coefs))
    classifier = RiskScoreClassifier(m, intercept, coefs, X_train = xtrain)

    ## We would want to supply the classifier with some out of sample data here.  I think
    ## that if we do, then the model will update its empirical risk estimates with the
    ## out of sample risk, but it's unclear from the examples.

    classifier.reset_featureNames(featurenames)
    #classifier.print_model_card()

    rslt_tbl = tabulate_scorecards(multipliers, beta0s, betas, featurenames)

    return(spark.createDataFrame(rslt_tbl))

def fr_prep_data(indata):
    indata = indata.dropna('any').toPandas()
    ytrain = np.asarray(indata.iloc[:, -1])       # outcome in the last column

    ## discretize the training variables
    features_discretized = convert_continuous_df_to_binary_df(indata.iloc[:, 0:-1])
    print('feature columns:\n')
    print(list(features_discretized.columns))
    print('len: ', len(list(features_discretized.columns)))

    xtrain = np.asarray(features_discretized)
    print(xtrain)
    print(xtrain.shape)

    featurenames = list(features_discretized.columns)

    return(xtrain, ytrain, featurenames)

def tabulate_scorecards(multipliers, beta0s, betas, featurenames):
    def scorecard_table(i):
        coefs = [multipliers[i], beta0s[i]] + list(betas[i])
        parms = ['(multiplier)', '(intercept)'] + featurenames
        cardidx = [i] * len(coefs)
        parmidx = range(len(coefs))

        tbl = pd.DataFrame({'cardid':cardidx, 'parmid':parmidx, 'parameter':parms, 'coefficient':coefs})

        ## Drop any beta coefficients that are zero (which should be most of them).
        ## Always keep multiplier and intercept
        eps = 1e-6              # Coefficients should be integers, but this saves us headaches if the zeros aren't exact.
        foo = list(abs(np.asarray(betas[i])))        
        keep = np.asarray([True, True] + list(abs(np.asarray(betas[i])) > eps))

        return tbl.loc[keep, :]

    return(pd.concat(map(scorecard_table, range(len(multipliers)))))

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e58c09a6-59a2-4751-9d1e-5d3460badb81"),
    test_scores=Input(rid="ri.vector.main.execute.a42ee3a8-0ccb-468e-bce0-457c73d7d12f")
)
def model_1a_score_test_preds(test_scores):
    risks = [0.2 , 0.2 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.8 , 1.0 , 1.3 , 1.7 , 2.1 , 2.7, 3.4, 4.2, 5.3, 6.7, 8.3, 10.3, 12.8, 15.7, 19.2, 23.2, 27.7]

    score_cols = [x for x in test_scores.columns if x != "hosp"]

    test_scores['total'] = test_scores[score_cols].sum(axis = 1)
    test_scores['pred'] = test_scores['total'].map(lambda x: risks[x+13] / 100.0)

    return(test_scores)

