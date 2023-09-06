

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0ff8c604-6ff4-43da-838c-0b6e31e86b80"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def Fit_final_model_1a(features):
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    X_return = pd.DataFrame()

    pred_cols = [x for x in features.columns if not x in ['hosp', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                            'data_partner_id','md_per_hundred_pop']]
    X_train = features[pred_cols]
    y_train = features['hosp']

    train_data = lgb.Dataset(X_train, label=y_train)
    param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
    param['metric'] = 'binary_logloss'
    num_round = 100
    bst = lgb.train(param, train_data, num_round)

    X_return = pd.DataFrame({
        'model': [bst.model_to_string()]
    })
    
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
    Output(rid="ri.vector.main.execute.eb796457-7aa7-470a-aad1-3fc2120d1789"),
    cv_pred=Input(rid="ri.foundry.main.dataset.101f94cc-ddda-447d-a94d-f081b4fb88df")
)
def cal_hist( cv_pred):
    test_predictions = cv_pred
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from sklearn.calibration import CalibrationDisplay
    import matplotlib
    print(matplotlib.__version__)
    #3.5.2

    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(8, 2)
    colors = plt.get_cmap("Dark2")
    plt.rcParams.update({'font.size': 12})
    bns = 20

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i in test_predictions.cv_group.unique():
        y_true =  test_predictions.loc[test_predictions.cv_group==i, ['hosp']]
        y_pred = test_predictions.loc[test_predictions.cv_group==i, ['pred']]
        name = 'fold ' + str(i)
        display = CalibrationDisplay.from_predictions(
            y_true,
            y_pred,
            n_bins=bns,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
            marker='o',
        )
        
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histogram
    grid_positions = [(x+2,y) for x in range(4) for y in range(2)] #[(2, 0), (2, 1), (3, 0), (3, 1)]
    for i in [2,3,4,5,6,7,8,9]:#test_predictions.cv_group.unique():
        row, col = grid_positions[i-2]
        ax = fig.add_subplot(gs[row, col])
        name = 'fold ' + str(i)
        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=bns,
            label=name,
            color=colors(i),
            log=True
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2a77db99-ab42-4b6d-be68-bcc025882778"),
    cv_pred_labs=Input(rid="ri.vector.main.execute.9e4e365b-557f-4d9b-b768-0ec89f834c08")
)
def cal_hist_labs( cv_pred_labs):
    test_predictions = cv_pred_labs
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from sklearn.calibration import CalibrationDisplay
    import matplotlib
    print(matplotlib.__version__)
    #3.5.2

    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(8, 2)
    colors = plt.get_cmap("Dark2")
    plt.rcParams.update({'font.size': 12})
    bns = 20

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i in test_predictions.cv_group.unique():
        y_true =  test_predictions.loc[test_predictions.cv_group==i, ['hosp']]
        y_pred = test_predictions.loc[test_predictions.cv_group==i, ['pred']]
        name = 'fold ' + str(i)
        display = CalibrationDisplay.from_predictions(
            y_true,
            y_pred,
            n_bins=bns,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
            marker='o',
        )
        
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histogram
    grid_positions = [(x+2,y) for x in range(4) for y in range(2)] #[(2, 0), (2, 1), (3, 0), (3, 1)]
    for i in [2,3,4,5,6,7,8,9]:#test_predictions.cv_group.unique():
        row, col = grid_positions[i-2]
        ax = fig.add_subplot(gs[row, col])
        name = 'fold ' + str(i)
        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=bns,
            label=name,
            color=colors(i),
            log=True
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.11991997-edd3-43d0-8b88-7196989e9612"),
    cv_pred=Input(rid="ri.foundry.main.dataset.101f94cc-ddda-447d-a94d-f081b4fb88df")
)
def calibration(cv_pred):
    test_predictions = cv_pred
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    for i in test_predictions.cv_group.unique():
        y_true =  test_predictions.loc[test_predictions.cv_group==i, ['hosp']]
        y_pred = test_predictions.loc[test_predictions.cv_group==i, ['pred']]
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
    Output(rid="ri.foundry.main.dataset.aaf168a5-6af7-47a1-9966-223daa10e10f"),
    cv_pred_labs=Input(rid="ri.vector.main.execute.9e4e365b-557f-4d9b-b768-0ec89f834c08")
)
def calibration_labs(cv_pred_labs):
    test_predictions = cv_pred_labs
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    for i in test_predictions.cv_group.unique():
        y_true =  test_predictions.loc[test_predictions.cv_group==i, ['hosp']]
        y_pred = test_predictions.loc[test_predictions.cv_group==i, ['pred']]
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
    Output(rid="ri.vector.main.execute.c3fba417-49c5-495e-aefb-78b0855ccd40"),
    final_model_1a_test_preds=Input(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161")
)
def calibration_test(final_model_1a_test_preds):
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    y_true =  final_model_1a_test_preds['hosp']
    y_pred = final_model_1a_test_preds['pred']
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20)
    plt.plot(prob_pred, prob_true, linestyle='--', marker='o')
    
    plt.xlim([0, 1])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(np.arange(0, 1.1, .1))
    plt.yticks(np.arange(0, 1.1, .1))
    plt.show()

    plt.hist(y_pred)
    plt.show()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.101f94cc-ddda-447d-a94d-f081b4fb88df"),
    cv_train=Input(rid="ri.foundry.main.dataset.0d7e31e8-287b-4fd1-921a-066d2ee3c76c"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def cv_pred(cv_train, features):
    test = cv_train
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()

    for i in features.cv_group.unique():
        X_test = features.loc[features.cv_group==i, ~features.columns.isin(['hosp', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                            'data_partner_id','md_per_hundred_pop'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features.cv_group==i, ['hosp']]
        model_string = str(test.loc[test.cv_group==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test = features.loc[features.cv_group==i]
        X_test['pred'] = ypred
        X_test['hosp'] = y_test
        X_test['cv_group'] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.vector.main.execute.9e4e365b-557f-4d9b-b768-0ec89f834c08"),
    cv_train_labs=Input(rid="ri.foundry.main.dataset.6cbb0f8b-7252-494c-9211-f2d7ad83da3c"),
    features_labs=Input(rid="ri.foundry.main.dataset.64ef334a-cb91-459f-9dec-cf646d71676c")
)
def cv_pred_labs(cv_train_labs, features_labs):
    test = cv_train_labs
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()

    for i in features_labs.cv_group.unique():
        X_test = features_labs.loc[features_labs.cv_group==i, ~features_labs.columns.isin(['hosp', 'person_id', 'cv_group'])]
        #X_test['sex'] = X_test['sex'].astype('category')
        #X_test['race'] = X_test['race'].astype('category')
        y_test = features_labs.loc[features_labs.cv_group==i, ['hosp']]
        model_string = str(test.loc[test.cv_group==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['hosp'] = y_test
        X_test['cv_group'] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.vector.main.execute.1b6508e9-6961-43f3-8fd4-81635f107137"),
    cv_train_val=Input(rid="ri.foundry.main.dataset.e31f9bf9-02d1-4272-aa52-25fd6081b3c0"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def cv_pred_val(cv_train_val, features):
    test = cv_train_val
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    X_return = pd.DataFrame()

    for i in features.cv_group.unique():
        X_test = features.loc[features.cv_group==i, ~features.columns.isin(['hosp', 'person_id', 'cv_group','days_since_pandemic','sex','race','covid_year',
                                                                            'data_partner_id'])]
        X_test['sex'] = X_test['sex'].astype('category')
        X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features.cv_group==i, ['hosp']]
        model_string = str(test.loc[test.cv_group==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['hosp'] = y_test
        X_test['cv_group'] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.vector.main.execute.5c75094c-e95f-481d-902e-81224d41ea3f"),
    cv_train=Input(rid="ri.foundry.main.dataset.0d7e31e8-287b-4fd1-921a-066d2ee3c76c"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def cv_shap(features, cv_train):
    test = cv_train
    import lightgbm as lgb
    import pandas as pd
    import shap

    for i in [5]:
        X_test = features.loc[features.cv_group==i, ~features.columns.isin(['hosp', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                            'data_partner_id','md_per_hundred_pop'])]
        y_test = features.loc[features.cv_group==i, ['hosp']]
        #print(test.loc[test.cv_group==i,'model_text'][0])
        model_string = str(test.loc[test.cv_group==i,'model_text'].iloc[0])
        #print(model_string)

        bst = lgb.Booster(model_str=model_string)
        bst.params['objective'] = 'binary'

        feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_test.columns})
        feature_imp = feature_imp.sort_values(by="Value", ascending=False)[0:15]

        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_test)
        
        for feat in feature_imp.Feature:
            shap.dependence_plot(feat, shap_values[1], X_test, xmin = "percentile(1)", xmax = "percentile(99)", alpha = 0.4)

@transform_pandas(
    Output(rid="ri.vector.main.execute.98cdd141-6c8b-41a4-901e-0516c74a28da"),
    cv_train_labs=Input(rid="ri.foundry.main.dataset.6cbb0f8b-7252-494c-9211-f2d7ad83da3c"),
    features_labs=Input(rid="ri.foundry.main.dataset.64ef334a-cb91-459f-9dec-cf646d71676c")
)
def cv_shap_labs( features_labs, cv_train_labs):
    test = cv_train_labs
    import lightgbm as lgb
    import pandas as pd
    import shap

    for i in [5]:
        X_test = features_labs.loc[features_labs.cv_group==i, ~features_labs.columns.isin(['hosp', 'person_id', 'cv_group'])]
        y_test = features_labs.loc[features_labs.cv_group==i, ['hosp']]
        #print(test.loc[test.cv_group==i,'model_text'][0])
        model_string = str(test.loc[test.cv_group==i,'model_text'].iloc[0])
        #print(model_string)

        bst = lgb.Booster(model_str=model_string)
        bst.params['objective'] = 'binary'

        feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_test.columns})
        feature_imp = feature_imp.sort_values(by="Value", ascending=False)[0:15]

        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_test)
        
        for feat in feature_imp.Feature:
            shap.dependence_plot(feat, shap_values[1], X_test, xmin = "percentile(1)", xmax = "percentile(99)", alpha = 0.4)

@transform_pandas(
    Output(rid="ri.vector.main.execute.277c8976-3516-4e77-bf96-0d19a0ab37f6"),
    cv_train_val=Input(rid="ri.foundry.main.dataset.e31f9bf9-02d1-4272-aa52-25fd6081b3c0"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def cv_shap_val(features, cv_train_val):
    test = cv_train_val
    import lightgbm as lgb
    import pandas as pd
    import shap

    for i in [5]:
        X_test = features.loc[features.cv_group==i, ~features.columns.isin(['hosp', 'person_id', 'cv_group'])]
        X_test['sex'] = X_test['sex'].astype('category')
        X_test['race'] = X_test['race'].astype('category')
        y_test = features.loc[features.cv_group==i, ['hosp']]
        #print(test.loc[test.cv_group==i,'model_text'][0])
        model_string = str(test.loc[test.cv_group==i,'model_text'].iloc[0])
        #print(model_string)

        bst = lgb.Booster(model_str=model_string)
        bst.params['objective'] = 'binary'

        feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_test.columns})
        feature_imp = feature_imp.sort_values(by="Value", ascending=False)[0:15]

        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_test)
        
        for feat in feature_imp.Feature:
            shap.dependence_plot(feat, shap_values[1], X_test, xmin = "percentile(1)", xmax = "percentile(99)", alpha = 0.4)

        

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0d7e31e8-287b-4fd1-921a-066d2ee3c76c"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)

def cv_train(features):
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features.cv_group.unique():
        X_train = features.loc[features.cv_group!=i, ~features.columns.isin(['hosp', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                             'data_partner_id','md_per_hundred_pop'])]
        #X_train['sex'] = X_train['sex'].astype('category')
        #X_train['race'] = X_train['race'].astype('category')
        y_train = features.loc[features.cv_group!=i, ['hosp']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
        param['metric'] = 'binary_logloss'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
    
        df = pd.DataFrame([[i, bst.model_to_string()]], columns=['cv_group','model_text'])

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
    Output(rid="ri.foundry.main.dataset.6cbb0f8b-7252-494c-9211-f2d7ad83da3c"),
    features_labs=Input(rid="ri.foundry.main.dataset.64ef334a-cb91-459f-9dec-cf646d71676c")
)

def cv_train_labs( features_labs):
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features_labs.cv_group.unique():
        X_train = features_labs.loc[features_labs.cv_group!=i, ~features_labs.columns.isin(['hosp', 'person_id', 'cv_group'])]
        #X_train['sex'] = X_train['sex'].astype('category')
        #X_train['race'] = X_train['race'].astype('category')
        y_train = features_labs.loc[features_labs.cv_group!=i, ['hosp']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
        param['metric'] = 'binary_logloss'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
    
        df = pd.DataFrame([[i, bst.model_to_string()]], columns=['cv_group','model_text'])

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
    Output(rid="ri.foundry.main.dataset.e31f9bf9-02d1-4272-aa52-25fd6081b3c0"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
def cv_train_val(features):
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features.cv_group.unique():
        X_train = features.loc[features.cv_group!=i, ~features.columns.isin(['hosp', 'person_id', 'cv_group'])]
        X_train['sex'] = X_train['sex'].astype('category')
        X_train['race'] = X_train['race'].astype('category')
        y_train = features.loc[features.cv_group!=i, ['hosp']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05, 'min_data_in_leaf':1000}
        param['metric'] = 'binary_logloss'
        num_round = 100
        bst = lgb.train(param, train_data, num_round)
    
        df = pd.DataFrame([[i, bst.model_to_string()]], columns=['cv_group','model_text'])

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
    Output(rid="ri.vector.main.execute.a0363927-59a4-4b89-b2e6-7d10ba67ab8c"),
    cv_pred=Input(rid="ri.foundry.main.dataset.101f94cc-ddda-447d-a94d-f081b4fb88df")
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

    for i in test_predictions.cv_group.unique():
        true = test_predictions.loc[test_predictions.cv_group==i]['hosp']
        pred = test_predictions.loc[test_predictions.cv_group==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions.cv_group!=i]['hosp'])
        base_rate = sum(outcome) * 1.0 / len(outcome)
        #base_rate = .001
        bs_ref = sum([np.square(base_rate - x) for x in list(true)]) * 1.0 / len(list(true))
        bs = sum(np.square(pred - true)) * 1.0 / len(list(true))
        bss = 1 - (bs / bs_ref)

        print(f"CV Fold: {i}")
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
    Output(rid="ri.vector.main.execute.a0ccd8fc-8edc-4929-b6e0-00ea1077f893"),
    cv_pred_labs=Input(rid="ri.vector.main.execute.9e4e365b-557f-4d9b-b768-0ec89f834c08")
)
def eval_labs( cv_pred_labs):
    test_predictions = cv_pred_labs
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])

    for i in test_predictions.cv_group.unique():
        true = test_predictions.loc[test_predictions.cv_group==i]['hosp']
        pred = test_predictions.loc[test_predictions.cv_group==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions.cv_group!=i]['hosp'])
        base_rate = sum(outcome) * 1.0 / len(outcome)
        #base_rate = .001
        bs_ref = sum([np.square(base_rate - x) for x in list(true)]) * 1.0 / len(list(true))
        bs = sum(np.square(pred - true)) * 1.0 / len(list(true))
        bss = 1 - (bs / bs_ref)

        print(f"CV Fold: {i}")
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
    Output(rid="ri.foundry.main.dataset.c3bb0837-8215-48bf-9bd4-433f7c09f4cf"),
    cv_pred=Input(rid="ri.foundry.main.dataset.101f94cc-ddda-447d-a94d-f081b4fb88df")
)
def eval_partner(cv_pred):
    test_predictions = cv_pred
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])
    col = 'data_partner_id'
    X_eval = pd.DataFrame()
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

        print(f"BSS: {bss}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        X_eval = X_eval.append(pd.DataFrame({col:[i],"BSS":[bss],'AUC':[roc_auc]}))
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.figure()
    X_eval.BSS.plot.hist( bins=100, alpha=0.5)
    plt.show()
    X_eval.AUC.plot.hist( bins=10, alpha=0.5)
    plt.show()
    return(X_eval)

@transform_pandas(
    Output(rid="ri.vector.main.execute.85583e8e-f978-4df4-972c-5a4c85632e66"),
    final_model_1a_test_preds=Input(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161")
)
def eval_partner_test(final_model_1a_test_preds):
    test_predictions = final_model_1a_test_preds
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])
    col = 'data_partner_id'
    X_eval = pd.DataFrame()
    print(test_predictions[col].unique())

    for i in test_predictions[col].unique():
        true = test_predictions.loc[test_predictions[col]==i]['hosp']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        #roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['hosp'])
        base_rate = sum(outcome) * 1.0 / len(outcome)
        #base_rate = .001
        bs_ref = sum([np.square(base_rate - x) for x in list(true)]) * 1.0 / len(list(true))
        bs = sum(np.square(pred - true)) * 1.0 / len(list(true))
        bss = 1 - (bs / bs_ref)

        print(f"BSS: {bss}")
        if all(true):
            roc_auc = 0
        else:
            fpr, tpr, thresholds = roc_curve(true,  pred)
            roc_auc = auc(fpr, tpr)
        X_eval = X_eval.append(pd.DataFrame({col:[i],"BSS":[bss],'AUC':[roc_auc]}))
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.figure()
    X_eval.BSS.plot.hist( bins=100, alpha=0.5)
    plt.show()
    X_eval.AUC.plot.hist( bins=10, alpha=0.5)
    plt.show()
    return(X_eval)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.825330aa-8369-434b-b0db-228f49259b96"),
    cv_pred=Input(rid="ri.foundry.main.dataset.101f94cc-ddda-447d-a94d-f081b4fb88df")
)
def eval_race(cv_pred):
    test_predictions = cv_pred
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])
    col = 'race'
    X_eval = pd.DataFrame()
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
        plt.title(f"{i}")
        X_eval = X_eval.append(pd.DataFrame({col:[i],"BSS":[bss]}))
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()
    plt.figure()
    X_eval.BSS.plot.hist( bins=10, alpha=0.5)
    plt.show()
    return(X_eval)

@transform_pandas(
    Output(rid="ri.vector.main.execute.3bc492db-268a-482f-bb72-93cdf0e28cbd"),
    final_model_1a_test_preds=Input(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161")
)
def eval_race_test(final_model_1a_test_preds):
    test_predictions = final_model_1a_test_preds
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    #train_test_split["hosp"] = pd.to_numeric(train_test_split["hosp"])
    #train_test_split["pred"] = pd.to_numeric(train_test_split["pred"])
    col = 'race'
    X_eval = pd.DataFrame()
    print(test_predictions[col].unique())

    fig, ax = plt.subplots(figsize=(10,6))

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
        print(f"rocauc: {roc:0.3f}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot(ax = ax, name = i)
        X_eval = X_eval.append(pd.DataFrame({col:[i],"BSS":[bss]}))
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()
    plt.figure()
    X_eval.BSS.plot.hist( bins=10, alpha=0.5)
    plt.show()
    return(X_eval)
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.0b403321-8b5d-479d-8fcc-5a3ad03208e0"),
    cv_pred=Input(rid="ri.foundry.main.dataset.101f94cc-ddda-447d-a94d-f081b4fb88df")
)
def eval_sex( cv_pred):
    test_predictions = cv_pred
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
        plt.title(f"{i}")
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.85818bbd-b2e9-4e2c-99c2-e76c0cdfe972"),
    final_model_1a_test_preds=Input(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161")
)
def eval_sex_test(final_model_1a_test_preds):
    test_predictions = final_model_1a_test_preds
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    col = 'sex'
    print(test_predictions[col].unique())

    for i in ['MALE','FEMALE']:
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
        print(f"rocauc: {roc:0.3f}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot()
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.1a252b88-b68d-423c-b280-9cfffb1c0ca0"),
    final_model_1a_test_preds=Input(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161")
)
def eval_test( final_model_1a_test_preds):

    from sklearn.metrics import brier_score_loss, roc_auc_score, auc, roc_curve
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    true =  final_model_1a_test_preds['hosp']
    pred = final_model_1a_test_preds['pred']
    
    bsl = brier_score_loss(true, pred)
    roc = roc_auc_score(true, pred)

    base_rate = sum(true) * 1.0 / len(true)

    bs_ref = sum([np.square(base_rate - x) for x in list(true)]) * 1.0 / len(list(true))
    bs = sum(np.square(pred - true)) * 1.0 / len(list(true))
    bss = 1 - (bs / bs_ref)

    print(f"brier: {bsl:0.4f}")
    print(f"BSS: {bss:0.4f}")
    print(f"rocauc: {roc:0.3f}")

    fpr, tpr, thresholds = roc_curve(true,  pred)
    roc_auc = auc(fpr, tpr)

    # RocCurveDisplay.from_predictions(true, pred)

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    RocCurveDisplay.from_predictions(true, pred, ax=ax[0], name='Test')
    PrecisionRecallDisplay.from_predictions(true, pred, ax=ax[1], name='Test')
    CalibrationDisplay.from_predictions(true, pred, ax=ax[2], name='Test', n_bins=20, strategy = 'quantile')
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.3f9c3efb-1046-4179-9882-b7f696436a67"),
    cv_pred=Input(rid="ri.foundry.main.dataset.101f94cc-ddda-447d-a94d-f081b4fb88df")
)
def eval_year(cv_pred):
    test_predictions = cv_pred
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
        plt.title(f"Year: {i}")
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.9a6447de-4ba6-463e-9405-fb2e6684f79d"),
    final_model_1a_test_preds=Input(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161")
)
def eval_year_test(final_model_1a_test_preds):
    test_predictions = final_model_1a_test_preds
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

    fig, ax = plt.subplots(figsize=(10,6))

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
        print(f"rocauc: {roc:0.3f}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot(ax = ax, name = i)
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e7468d12-abc8-43bd-86b1-6c1e92d06277"),
    Fit_final_model_1a=Input(rid="ri.foundry.main.dataset.0ff8c604-6ff4-43da-838c-0b6e31e86b80"),
    final_model_1a_test_preds=Input(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161")
)
def final_model_1a_shap(final_model_1a_test_preds, Fit_final_model_1a):
    import lightgbm as lgb
    import pandas as pd
    import shap
    
    pred_cols = [x for x in final_model_1a_test_preds.columns if not x in ['hosp', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                    'data_partner_id','md_per_hundred_pop', 'pred']]
    X_test = final_model_1a_test_preds[pred_cols]
        
    model_string = str(Fit_final_model_1a.iloc[0,0])

    bst = lgb.Booster(model_str=model_string)
    bst.params['objective'] = 'binary'

    feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_test.columns})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False)

    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_test)
        
    for feat in feature_imp.Feature:
        shap.dependence_plot(feat, shap_values[1], X_test, xmin = "percentile(1)", xmax = "percentile(99)", alpha = 0.4)

    ddf = pd.DataFrame(shap_values[1])
    ddf.columns = [f"{x}_shap" for x in X_test.columns]
    for x in X_test.columns:
        ddf[x] = X_test[x].values

    ddf['age'] = ddf['age'].astype('int32')
    
    return(ddf)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161"),
    Fit_final_model_1a=Input(rid="ri.foundry.main.dataset.0ff8c604-6ff4-43da-838c-0b6e31e86b80"),
    features_test=Input(rid="ri.foundry.main.dataset.e46f083a-ec97-4db9-bdcd-9e344d8528dc")
)
def final_model_1a_test_preds(Fit_final_model_1a, features_test):
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    pred_cols = [x for x in features_test.columns if not x in ['hosp', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                            'data_partner_id','md_per_hundred_pop']]
    X_test = features_test[pred_cols]
    model_string = str(Fit_final_model_1a.iloc[0,0])

    bst = lgb.Booster(model_str=model_string)
    ypred = bst.predict(X_test)

    features_test['pred'] = ypred
    return features_test

