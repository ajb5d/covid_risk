

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.fc51f3a9-f34a-4219-82a6-f278892d9259"),
    Features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def CV_race(Features):
    features = Features
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()
    col = 'race'

    for i in features[col].unique():
        X_train = features.loc[features[col]!=i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group', col, 'sex','covid_year','years_since_pandemic','data_partner_id'])]
        y_train = features.loc[features[col]!=i, ['composite_outcome']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
        param['metric'] = 'binary_logloss'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
    
        df = pd.DataFrame([[i, bst.model_to_string()]], columns=[col,'model_text'])

        X_return = pd.concat([X_return, df])

    return(X_return)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c9653ebe-49ab-4302-8dc6-5027d3b94b81"),
    Features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def CV_year(Features):
    features = Features
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()
    col = 'covid_year'

    for i in features[col].unique():
        X_train = features.loc[features[col]!=i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group', col, 'sex','race','years_since_pandemic','data_partner_id'])]
        y_train = features.loc[features[col]!=i, ['composite_outcome']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
        param['metric'] = 'binary_logloss'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
    
        df = pd.DataFrame([[i, bst.model_to_string()]], columns=[col,'model_text'])

        X_return = pd.concat([X_return, df])

        import matplotlib.pyplot as plt
        import seaborn as sns

        feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_train.columns})
        fig_size = (40, 20)
        plt.figure(figsize=fig_size)
        sns.set(font_scale = 5)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                            ascending=False)[0:15])
        plt.title('LightGBM Features')
        plt.tight_layout()
        plt.show()
    
    return(X_return)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.78ccdb8e-259c-4f5a-811d-d10411aa00bc"),
    CV_race=Input(rid="ri.foundry.main.dataset.fc51f3a9-f34a-4219-82a6-f278892d9259"),
    Features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def Cv_pred_race(CV_race, Features):
    test = CV_race
    features = Features
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()
    #part_ids = [x for x in features.data_partner_id.unique() if str(x) != 'nan']
    col = 'race'
    print(test[col].unique())

    for i in features[col].unique():
        X_test = features.loc[features[col]==i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group', col, 'sex','covid_year','years_since_pandemic','data_partner_id'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features[col]==i, ['composite_outcome']]
        model_string = str(test.loc[test[col]==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['composite_outcome'] = y_test
        X_test[col] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cafd47ad-e2fa-49e6-8e04-e3a5b30da899"),
    Cv_sex=Input(rid="ri.foundry.main.dataset.42ccae3b-0c96-40a4-887a-39916fdeb833"),
    Features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def Cv_pred_sex( Features, Cv_sex):
    test = Cv_sex
    features = Features
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()
    #part_ids = [x for x in features.data_partner_id.unique() if str(x) != 'nan']
    col = 'sex'
    print(test[col].unique())

    for i in features[col].unique():
        X_test = features.loc[features[col]==i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group', col, 'race','covid_year','years_since_pandemic','data_partner_id'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features[col]==i, ['composite_outcome']]
        model_string = str(test.loc[test[col]==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['composite_outcome'] = y_test
        X_test[col] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f3d49437-9cd7-458d-aa04-0f197f8ce70c"),
    CV_year=Input(rid="ri.foundry.main.dataset.c9653ebe-49ab-4302-8dc6-5027d3b94b81"),
    Features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def Cv_pred_year(CV_year, Features):
    test = CV_year
    features = Features
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()
    #part_ids = [x for x in features.data_partner_id.unique() if str(x) != 'nan']
    col = 'covid_year'
    print(test[col].unique())

    for i in features[col].unique():
        X_test = features.loc[features[col]==i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group', col, 'sex','race','years_since_pandemic','data_partner_id'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features[col]==i, ['composite_outcome']]
        model_string = str(test.loc[test[col]==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['composite_outcome'] = y_test
        X_test[col] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.42ccae3b-0c96-40a4-887a-39916fdeb833"),
    Features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)

def Cv_sex(Features):
    features = Features
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()
    col = 'sex'

    for i in features[col].unique():
        X_train = features.loc[features[col]!=i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group', col, 'race','covid_year','years_since_pandemic','data_partner_id'])]
        #X_train['sex'] = X_train['sex'].astype('category')
        #X_train['race'] = X_train['race'].astype('category')
        y_train = features.loc[features[col]!=i, ['composite_outcome']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
        param['metric'] = 'binary_logloss'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
    
        df = pd.DataFrame([[i, bst.model_to_string()]], columns=[col,'model_text'])

        X_return = pd.concat([X_return, df])

        #import matplotlib.pyplot as plt
        #import seaborn as sns

        #feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_train.columns})
        #fig_size = (40, 20)
        #plt.figure(figsize=fig_size)
        #sns.set(font_scale = 5)
        #sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
        #                                                    ascending=False)[0:15])
        #plt.title('LightGBM Features')
        #plt.tight_layout()
        #plt.show()
    
    return(X_return)

@transform_pandas(
    Output(rid="ri.vector.main.execute.748b6f21-1ac1-4ee5-931d-5d43b79aaf3e"),
    cv_pred=Input(rid="ri.foundry.main.dataset.46d81741-24b4-4761-9fe1-ffc24883c2d4")
)
def calibration(cv_pred):
    test_predictions = cv_pred
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    for i in [124,399,569]:#test_predictions.data_partner_id.unique()[10:20]:
        y_true =  test_predictions.loc[test_predictions.data_partner_id==i, ['composite_outcome']]
        y_pred = test_predictions.loc[test_predictions.data_partner_id==i, ['pred']]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
        plt.plot(prob_pred, prob_true, linestyle='--', marker='o')
        #disp = CalibrationDisplay.from_predictions(y_true, y_pred)
    plt.xlim([0, 1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.855993d8-3072-48b6-a247-8bf9141be6b5"),
    Cv_pred_race=Input(rid="ri.foundry.main.dataset.78ccdb8e-259c-4f5a-811d-d10411aa00bc")
)
def calibration_race(Cv_pred_race):
    test_predictions = Cv_pred_race
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    col = 'race'

    for i in test_predictions[col].unique():
        y_true =  test_predictions.loc[test_predictions[col]==i, ['composite_outcome']]
        y_pred = test_predictions.loc[test_predictions[col]==i, ['pred']]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
        plt.plot(prob_pred, prob_true, linestyle='--', marker='o')
        #disp = CalibrationDisplay.from_predictions(y_true, y_pred)
    plt.xlim([0, 1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.2d2a4740-7a45-49fc-a0b5-226813e62be1"),
    Cv_pred_sex=Input(rid="ri.foundry.main.dataset.cafd47ad-e2fa-49e6-8e04-e3a5b30da899")
)
def calibration_sex(Cv_pred_sex):
    test_predictions = Cv_pred_sex
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    col = 'sex'

    for i in test_predictions[col].unique():
        y_true =  test_predictions.loc[test_predictions[col]==i, ['composite_outcome']]
        y_pred = test_predictions.loc[test_predictions[col]==i, ['pred']]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
        plt.plot(prob_pred, prob_true, linestyle='--', marker='o')
        #disp = CalibrationDisplay.from_predictions(y_true, y_pred)
    plt.xlim([0, 1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.364c56b8-98c8-4bf3-a040-cf4b46e6d94a"),
    Cv_pred_year=Input(rid="ri.foundry.main.dataset.f3d49437-9cd7-458d-aa04-0f197f8ce70c")
)
def calibration_year(Cv_pred_year):
    test_predictions = Cv_pred_year
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    col = 'covid_year'

    for i in test_predictions[col].unique():
        y_true =  test_predictions.loc[test_predictions[col]==i, ['composite_outcome']]
        y_pred = test_predictions.loc[test_predictions[col]==i, ['pred']]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
        plt.plot(prob_pred, prob_true, linestyle='--', marker='o')
        #disp = CalibrationDisplay.from_predictions(y_true, y_pred)
    plt.xlim([0, 1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.show()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7ace7e9f-c314-41be-9546-8274cf3ece41"),
    Features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def cv_partner(Features):
    features = Features
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features.data_partner_id.unique():
        X_train = features.loc[features.data_partner_id!=i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group', 'data_partner_id', 'race','sex','covid_year','years_since_pandemic'])]
        #X_train['sex'] = X_train['sex'].astype('category')
        #X_train['race'] = X_train['race'].astype('category')
        y_train = features.loc[features.data_partner_id!=i, ['composite_outcome']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
        param['metric'] = 'binary_logloss'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
    
        df = pd.DataFrame([[i, bst.model_to_string()]], columns=['data_partner_id','model_text'])

        X_return = pd.concat([X_return, df])

        #import matplotlib.pyplot as plt
        #import seaborn as sns

        #feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_train.columns})
        #fig_size = (40, 20)
        #plt.figure(figsize=fig_size)
        #sns.set(font_scale = 5)
        #sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
        #                                                    ascending=False)[0:15])
        #plt.title('LightGBM Features')
        #plt.tight_layout()
        #plt.show()
    
    return(X_return)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.46d81741-24b4-4761-9fe1-ffc24883c2d4"),
    Features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6"),
    cv_partner=Input(rid="ri.foundry.main.dataset.7ace7e9f-c314-41be-9546-8274cf3ece41")
)
def cv_pred( Features, cv_partner):
    test = cv_partner
    features = Features
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()
    part_ids = [x for x in features.data_partner_id.unique() if str(x) != 'nan']

    for i in part_ids:
        X_test = features.loc[features.data_partner_id==i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group','data_partner_id', 'race','sex','covid_year','years_since_pandemic'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features.data_partner_id==i, ['composite_outcome']]
        model_string = str(test.loc[test.data_partner_id==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['composite_outcome'] = y_test
        X_test['data_partner_id'] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.vector.main.execute.f164ebeb-b7a6-41d3-85a5-89d1e1a55e8e"),
    cv_pred=Input(rid="ri.foundry.main.dataset.46d81741-24b4-4761-9fe1-ffc24883c2d4")
)
def eval( cv_pred):
    test_predictions = cv_pred
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])

    for i in test_predictions.data_partner_id.unique():
        true = test_predictions.loc[test_predictions.data_partner_id==i]['composite_outcome']
        pred = test_predictions.loc[test_predictions.data_partner_id==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions.data_partner_id!=i]['composite_outcome'])
        base_rate = sum(outcome) * 1.0 / len(outcome)
        #base_rate = .001
        bs_ref = sum([np.square(base_rate - x) for x in list(true)]) * 1.0 / len(list(true))
        bs = sum(np.square(pred - true)) * 1.0 / len(list(true))
        bss = 1 - (bs / bs_ref)

        print(f"Prtner ID: {i}")
        print(f"brier: {bsl}")
        print(f"BSS: {bss}")
        print(f"rocauc: {roc}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot()
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.082e5147-2bf1-463b-ade9-7cf30a429373"),
    Cv_pred_race=Input(rid="ri.foundry.main.dataset.78ccdb8e-259c-4f5a-811d-d10411aa00bc")
)
def eval_race(Cv_pred_race):
    test_predictions = Cv_pred_race
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])
    col = 'race'
    print(test_predictions[col].unique())

    for i in test_predictions[col].unique():
        true = test_predictions.loc[test_predictions[col]==i]['composite_outcome']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['composite_outcome'])
        base_rate = sum(outcome) * 1.0 / len(outcome)
        #base_rate = .001
        bs_ref = sum([np.square(base_rate - x) for x in list(true)]) * 1.0 / len(list(true))
        bs = sum(np.square(pred - true)) * 1.0 / len(list(true))
        bss = 1 - (bs / bs_ref)

        print(f"{col}: {i}")
        print(f"brier: {bsl}")
        print(f"BSS: {bss}")
        print(f"rocauc: {roc}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot()
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.832e3651-7551-4f61-a784-e0eca05bc311"),
    Cv_pred_sex=Input(rid="ri.foundry.main.dataset.cafd47ad-e2fa-49e6-8e04-e3a5b30da899")
)
def eval_sex( Cv_pred_sex):
    test_predictions = Cv_pred_sex
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])
    col = 'sex'
    print(test_predictions[col].unique())

    for i in test_predictions[col].unique():
        true = test_predictions.loc[test_predictions[col]==i]['composite_outcome']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['composite_outcome'])
        base_rate = sum(outcome) * 1.0 / len(outcome)
        #base_rate = .001
        bs_ref = sum([np.square(base_rate - x) for x in list(true)]) * 1.0 / len(list(true))
        bs = sum(np.square(pred - true)) * 1.0 / len(list(true))
        bss = 1 - (bs / bs_ref)

        print(f"{col}: {i}")
        print(f"brier: {bsl}")
        print(f"BSS: {bss}")
        print(f"rocauc: {roc}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot()
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.79c9965d-0edd-4981-bf44-df5cd2a89346"),
    Cv_pred_year=Input(rid="ri.foundry.main.dataset.f3d49437-9cd7-458d-aa04-0f197f8ce70c")
)
def eval_year(Cv_pred_year):
    test_predictions = Cv_pred_year
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])
    col = 'covid_year'
    print(test_predictions[col].unique())

    for i in test_predictions[col].unique():
        true = test_predictions.loc[test_predictions[col]==i]['composite_outcome']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['composite_outcome'])
        base_rate = sum(outcome) * 1.0 / len(outcome)
        #base_rate = .001
        bs_ref = sum([np.square(base_rate - x) for x in list(true)]) * 1.0 / len(list(true))
        bs = sum(np.square(pred - true)) * 1.0 / len(list(true))
        bss = 1 - (bs / bs_ref)

        print(f"{col}: {i}")
        print(f"brier: {bsl}")
        print(f"BSS: {bss}")
        print(f"rocauc: {roc}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot()
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

