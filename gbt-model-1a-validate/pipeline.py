

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.33d0caf6-5ff6-4073-9641-c092a63a51a7"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def CV_race(features):
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()
    col = 'race'

    for i in features[col].unique():
        X_train = features.loc[features[col]!=i, ~features.columns.isin(['hosp', 'person_id', 'cv_group', col, 'sex','covid_year','years_since_pandemic'])]
        y_train = features.loc[features[col]!=i, ['hosp']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
        param['metric'] = 'binary_logloss'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
    
        df = pd.DataFrame([[i, bst.model_to_string()]], columns=[col,'model_text'])

        X_return = pd.concat([X_return, df])

    return(X_return)
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.34e8b5c2-8a1c-44ce-b876-3076940332a6"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def CV_year(features):
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()
    col = 'covid_year'

    for i in features[col].unique():
        X_train = features.loc[features[col]!=i, ~features.columns.isin(['hosp', 'person_id', 'cv_group', col, 'sex','race','years_since_pandemic','data_partner_id'])]
        y_train = features.loc[features[col]!=i, ['hosp']]

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
    Output(rid="ri.foundry.main.dataset.8651ea6e-f25a-4dc5-b5eb-51abb20ba794"),
    CV_race=Input(rid="ri.foundry.main.dataset.33d0caf6-5ff6-4073-9641-c092a63a51a7"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def Cv_pred_race(CV_race, features):
    test = CV_race
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()
    #part_ids = [x for x in features.data_partner_id.unique() if str(x) != 'nan']
    col = 'race'
    print(test[col].unique())

    for i in features[col].unique():
        X_test = features.loc[features[col]==i, ~features.columns.isin(['hosp', 'person_id', 'cv_group',col,'sex','covid_year','years_since_pandemic'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features[col]==i, ['hosp']]
        model_string = str(test.loc[test[col]==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['hosp'] = y_test
        X_test[col] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.50da6785-9a1a-429e-bb10-0919c83a6929"),
    Cv_sex=Input(rid="ri.foundry.main.dataset.8f75f2df-8c88-4662-a51e-35fc9d12006a"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def Cv_pred_sex( features, Cv_sex):
    test = Cv_sex
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()
    #part_ids = [x for x in features.data_partner_id.unique() if str(x) != 'nan']
    col = 'sex'
    print(test[col].unique())

    for i in features[col].unique():
        X_test = features.loc[features[col]==i, ~features.columns.isin(['hosp', 'person_id', 'cv_group',col,'race','covid_year','years_since_pandemic'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features[col]==i, ['hosp']]
        model_string = str(test.loc[test[col]==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['hosp'] = y_test
        X_test[col] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b1c7eab0-7b4d-4750-8738-871aa884f704"),
    CV_year=Input(rid="ri.foundry.main.dataset.34e8b5c2-8a1c-44ce-b876-3076940332a6"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def Cv_pred_year(CV_year, features):
    test = CV_year
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()
    #part_ids = [x for x in features.data_partner_id.unique() if str(x) != 'nan']
    col = 'covid_year'
    print(test[col].unique())

    for i in features[col].unique():
        X_test = features.loc[features[col]==i, ~features.columns.isin(['hosp', 'person_id', 'cv_group', col, 'sex','race','years_since_pandemic','data_partner_id'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features[col]==i, ['hosp']]
        model_string = str(test.loc[test[col]==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['hosp'] = y_test
        X_test[col] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8f75f2df-8c88-4662-a51e-35fc9d12006a"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)

def Cv_sex(features):
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()
    col = 'sex'

    for i in features[col].unique():
        X_train = features.loc[features[col]!=i, ~features.columns.isin(['hosp', 'person_id', 'cv_group', col, 'race','covid_year','years_since_pandemic'])]
        #X_train['sex'] = X_train['sex'].astype('category')
        #X_train['race'] = X_train['race'].astype('category')
        y_train = features.loc[features[col]!=i, ['hosp']]

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
    Output(rid="ri.vector.main.execute.89d5af05-18bb-41b4-b872-cb4b5d93b132"),
    cv_pred=Input(rid="ri.foundry.main.dataset.27427104-8081-437a-8508-e85d970ae634")
)
def calibration(cv_pred):
    test_predictions = cv_pred
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    for i in [124,399,569]:#test_predictions.data_partner_id.unique()[10:20]:
        y_true =  test_predictions.loc[test_predictions.data_partner_id==i, ['hosp']]
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
    Output(rid="ri.vector.main.execute.f3fcdd22-28de-4d10-aacb-dc6358e979c4"),
    Cv_pred_race=Input(rid="ri.foundry.main.dataset.8651ea6e-f25a-4dc5-b5eb-51abb20ba794")
)
def calibration_race(Cv_pred_race):
    test_predictions = Cv_pred_race
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    col = 'race'

    for i in test_predictions[col].unique():
        y_true =  test_predictions.loc[test_predictions[col]==i, ['hosp']]
        y_pred = test_predictions.loc[test_predictions[col]==i, ['pred']]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
        plt.plot(prob_pred, prob_true, linestyle='--', marker='o', label = i)
        #disp = CalibrationDisplay.from_predictions(y_true, y_pred)
    plt.xlim([0, 1])
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.ec61a947-2400-4e47-a490-206a740f6d01"),
    Cv_pred_sex=Input(rid="ri.foundry.main.dataset.50da6785-9a1a-429e-bb10-0919c83a6929")
)
def calibration_sex(Cv_pred_sex):
    test_predictions = Cv_pred_sex
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    col = 'sex'

    for i in test_predictions[col].unique():
        y_true =  test_predictions.loc[test_predictions[col]==i, ['hosp']]
        y_pred = test_predictions.loc[test_predictions[col]==i, ['pred']]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
        plt.plot(prob_pred, prob_true, linestyle='--', marker='o', label = i)
        #disp = CalibrationDisplay.from_predictions(y_true, y_pred)
    plt.legend()
    plt.xlim([0, 1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.9386ebce-adb5-4047-8a2d-a8e59d4080c4"),
    Cv_pred_year=Input(rid="ri.foundry.main.dataset.b1c7eab0-7b4d-4750-8738-871aa884f704")
)
def calibration_year(Cv_pred_year):
    test_predictions = Cv_pred_year
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    col = 'covid_year'

    for i in test_predictions[col].unique():
        y_true =  test_predictions.loc[test_predictions[col]==i, ['hosp']]
        y_pred = test_predictions.loc[test_predictions[col]==i, ['pred']]
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
        plt.plot(prob_pred, prob_true, linestyle='--', marker='o', label = i)
        #disp = CalibrationDisplay.from_predictions(y_true, y_pred)
    plt.xlim([0, 1])
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.show()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0dcd66f4-9d0e-427e-be2a-cf2a9583591e"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)

def cv_partner(features):
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features.data_partner_id.unique():
        X_train = features.loc[features.data_partner_id!=i, ~features.columns.isin(['hosp', 'person_id', 'cv_group', 'data_partner_id', 'race','sex','covid_year','years_since_pandemic'])]
        #X_train['sex'] = X_train['sex'].astype('category')
        #X_train['race'] = X_train['race'].astype('category')
        y_train = features.loc[features.data_partner_id!=i, ['hosp']]

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
    Output(rid="ri.foundry.main.dataset.27427104-8081-437a-8508-e85d970ae634"),
    cv_partner=Input(rid="ri.foundry.main.dataset.0dcd66f4-9d0e-427e-be2a-cf2a9583591e"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def cv_pred( features, cv_partner):
    test = cv_partner
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()
    part_ids = [x for x in features.data_partner_id.unique() if str(x) != 'nan']

    for i in part_ids:
        X_test = features.loc[features.data_partner_id==i, ~features.columns.isin(['hosp', 'person_id', 'cv_group','data_partner_id', 'race','sex','covid_year','years_since_pandemic'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features.data_partner_id==i, ['hosp']]
        model_string = str(test.loc[test.data_partner_id==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['hosp'] = y_test
        X_test['data_partner_id'] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.vector.main.execute.3beb0c44-74c2-421c-ba27-24b06dcf9dd7"),
    cv_pred=Input(rid="ri.foundry.main.dataset.27427104-8081-437a-8508-e85d970ae634")
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
        true = test_predictions.loc[test_predictions.data_partner_id==i]['hosp']
        pred = test_predictions.loc[test_predictions.data_partner_id==i]['pred']
        bsl = brier_score_loss(true, pred)
        if not all(true):
            roc = roc_auc_score(true, pred)
        else:
            roc = 0

        outcome = list(test_predictions.loc[test_predictions.data_partner_id!=i]['hosp'])
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
    Output(rid="ri.vector.main.execute.dec99864-fda4-4708-a602-ccf56922e3bf"),
    Cv_pred_race=Input(rid="ri.foundry.main.dataset.8651ea6e-f25a-4dc5-b5eb-51abb20ba794")
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
        true = test_predictions.loc[test_predictions[col]==i]['hosp']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['hosp'])
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
    Output(rid="ri.vector.main.execute.56ce4a48-a33a-47fa-8a3e-aa41e302a1f3"),
    Cv_pred_sex=Input(rid="ri.foundry.main.dataset.50da6785-9a1a-429e-bb10-0919c83a6929")
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
        true = test_predictions.loc[test_predictions[col]==i]['hosp']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['hosp'])
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
    Output(rid="ri.vector.main.execute.60df343a-4c6d-4fa4-b247-2b68cb05b6ca"),
    Cv_pred_year=Input(rid="ri.foundry.main.dataset.b1c7eab0-7b4d-4750-8738-871aa884f704")
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
        true = test_predictions.loc[test_predictions[col]==i]['hosp']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['hosp'])
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

