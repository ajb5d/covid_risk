

@transform_pandas(
    Output(rid="ri.vector.main.execute.4bcb504f-dc80-460c-8aa1-e7c8b3c80d43"),
    cv_pred=Input(rid="ri.foundry.main.dataset.e6b6df1b-4b69-471f-89fe-aca516a015ab")
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
        y_true =  test_predictions.loc[test_predictions.cv_group==i, ['composite_outcome']]
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
            log=False
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.b52ab438-babd-43c8-a23c-a5982a26356c"),
    cv_pred=Input(rid="ri.foundry.main.dataset.e6b6df1b-4b69-471f-89fe-aca516a015ab")
)
def calibration(cv_pred):
    test_predictions = cv_pred
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    for i in test_predictions.cv_group.unique():
        y_true =  test_predictions.loc[test_predictions.cv_group==i, ['composite_outcome']]
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
    Output(rid="ri.vector.main.execute.0268f2bd-06b2-443c-86eb-bc8ece4cbd9f"),
    final_model_1b_test_preds=Input(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea")
)
def calibration_test(final_model_1b_test_preds):
    test_predictions = final_model_1b_test_preds
    from sklearn.calibration import calibration_curve
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np

    y_true =  test_predictions['composite_outcome']
    y_pred = test_predictions['pred']
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
    Output(rid="ri.foundry.main.dataset.e6b6df1b-4b69-471f-89fe-aca516a015ab"),
    cv_train=Input(rid="ri.foundry.main.dataset.e65da88d-c182-47c3-a0a9-8e4af26c4d27"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def cv_pred(cv_train, features):
    model = cv_train
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features.cv_group.unique():
        X_test = features.loc[features.cv_group==i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                             'data_partner_id'])]
        y_test = features.loc[features.cv_group==i, ['composite_outcome']]
        model_string = str(model.loc[model.cv_group==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test = features.loc[features.cv_group==i]
        X_test['pred'] = ypred
        X_test['composite_outcome'] = y_test
        X_test['cv_group'] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.vector.main.execute.91aef906-1fbb-438f-a987-915bb1c3fce0"),
    cv_train_val=Input(rid="ri.foundry.main.dataset.3419b1e1-3019-4419-88f4-aa9d1ac4e8b6"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def cv_pred_val(cv_train_val, features):
    model = cv_train_val
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features.cv_group.unique():
        X_test = features.loc[features.cv_group==i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                             'data_partner_id'])]
        y_test = features.loc[features.cv_group==i, ['composite_outcome']]
        X_test['sex'] = X_test['sex'].astype('category')
        X_test['race'] = X_test['race'].astype('category')

        model_string = str(model.loc[model.cv_group==i,'model_text'].iloc[0])

        bst = lgb.Booster(model_str=model_string)
        ypred = bst.predict(X_test)
        X_test['pred'] = ypred
        X_test['hosp'] = y_test
        X_test['cv_group'] = i

        X_return = pd.concat([X_return, X_test])
    return(X_return)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dd0ec65e-7c2e-48a8-9a17-d29370ef2ae7"),
    cv_train=Input(rid="ri.foundry.main.dataset.e65da88d-c182-47c3-a0a9-8e4af26c4d27"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def cv_shap( features, cv_train):
    model = cv_train
    import lightgbm as lgb
    import pandas as pd
    import shap

    for i in [5]:
        X_test = features.loc[features.cv_group==i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                             'data_partner_id'])]
        y_test = features.loc[features.cv_group==i, ['composite_outcome']]
        #print(test.loc[test.cv_group==i,'model_text'][0])
        model_string = str(model.loc[model.cv_group==i,'model_text'].iloc[0])
        #print(model_string)

        bst = lgb.Booster(model_str=model_string)
        bst.params['objective'] = 'binary'

        feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_test.columns})
        feature_imp = feature_imp.sort_values(by="Value", ascending=False)[0:20]

        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_test)
        
        for feat in feature_imp.Feature:
            shap.dependence_plot(feat, shap_values[1], X_test,
                xmin = "percentile(1)", xmax = "percentile(99)", alpha = 0.4)
        shap.dependence_plot('ddimer', shap_values[1], X_test,
                xmin = "percentile(1)", xmax = "percentile(99)", alpha = 0.4)

    imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_test.columns})
    print(imp.sort_values(by="Value", ascending=False)[0:15])

@transform_pandas(
    Output(rid="ri.vector.main.execute.be122f0b-6567-4cbf-8784-18896bb09aa1"),
    cv_train_val=Input(rid="ri.foundry.main.dataset.3419b1e1-3019-4419-88f4-aa9d1ac4e8b6"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def cv_shap_val( features, cv_train_val):
    model = cv_train_val
    import lightgbm as lgb
    import pandas as pd
    import shap

    for i in [5]:
        X_test = features.loc[features.cv_group==i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group'])]
        y_test = features.loc[features.cv_group==i, ['composite_outcome']]
        X_test['sex'] = X_test['sex'].astype('category')
        X_test['race'] = X_test['race'].astype('category')
        #print(test.loc[test.cv_group==i,'model_text'][0])
        model_string = str(model.loc[model.cv_group==i,'model_text'].iloc[0])
        #print(model_string)

        bst = lgb.Booster(model_str=model_string)
        bst.params['objective'] = 'binary'

        feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_test.columns})
        feature_imp = feature_imp.sort_values(by="Value", ascending=False)[0:10]

        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(X_test)
        
        for feat in feature_imp.Feature:
            shap.dependence_plot(feat, shap_values[1], X_test,
                xmin = "percentile(1)", xmax = "percentile(99)", alpha = 0.4)
        #shap.dependence_plot('age', shap_values[1], X_test)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e65da88d-c182-47c3-a0a9-8e4af26c4d27"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)

def cv_train( features):
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features.cv_group.unique():
        X_train = features.loc[features.cv_group!=i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                             'data_partner_id'])]
        y_train = features.loc[features.cv_group!=i, ['composite_outcome']]

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05}
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
    Output(rid="ri.foundry.main.dataset.3419b1e1-3019-4419-88f4-aa9d1ac4e8b6"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def cv_train_val( features):
    import lightgbm as lgb
    import pandas as pd

    X_return = pd.DataFrame()

    for i in features.cv_group.unique():
        X_train = features.loc[features.cv_group!=i, ~features.columns.isin(['composite_outcome', 'person_id', 'cv_group'])]
        y_train = features.loc[features.cv_group!=i, ['composite_outcome']]
        X_train['sex'] = X_train['sex'].astype('category')
        X_train['race'] = X_train['race'].astype('category')

        train_data = lgb.Dataset(X_train, label=y_train)
        param = {'objective': 'binary', 'learning_rate':0.05}
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
    Output(rid="ri.vector.main.execute.a318a285-2453-4e59-ba69-5ebdff6c8a7d"),
    cv_pred=Input(rid="ri.foundry.main.dataset.e6b6df1b-4b69-471f-89fe-aca516a015ab")
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
        true = test_predictions.loc[test_predictions.cv_group==i]['composite_outcome']
        pred = test_predictions.loc[test_predictions.cv_group==i]['pred']
        bsl = brier_score_loss(true, pred)
        roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions.cv_group!=i]['composite_outcome'])
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
    Output(rid="ri.foundry.main.dataset.a3c0e4f8-2cdc-4dfa-98d9-41c0249fdde9"),
    cv_pred=Input(rid="ri.foundry.main.dataset.e6b6df1b-4b69-471f-89fe-aca516a015ab")
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
        true = test_predictions.loc[test_predictions[col]==i]['composite_outcome']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        #roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['composite_outcome'])
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
    Output(rid="ri.foundry.main.dataset.511a487e-d7ff-4525-8f9d-c8b865f9c104"),
    final_model_1b_test_preds=Input(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea")
)
def eval_partner_test(final_model_1b_test_preds):
    test_predictions = final_model_1b_test_preds
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
        true = test_predictions.loc[test_predictions[col]==i]['composite_outcome']
        pred = test_predictions.loc[test_predictions[col]==i]['pred']
        bsl = brier_score_loss(true, pred)
        #roc = roc_auc_score(true, pred)

        outcome = list(test_predictions.loc[test_predictions[col]!=i]['composite_outcome'])
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
        X_eval = X_eval.append(pd.DataFrame({col:[i],"BSS":[bss],'AUC':[roc_auc], 'n': len(true)}))
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.figure()
    X_eval.BSS.plot.hist( bins=100, alpha=0.5)
    plt.show()
    X_eval.AUC.plot.hist( bins=10, alpha=0.5)
    plt.show()
    return(X_eval)

@transform_pandas(
    Output(rid="ri.vector.main.execute.91c6e9a7-3476-42a5-b6cc-191099a9bb7c"),
    cv_pred=Input(rid="ri.foundry.main.dataset.e6b6df1b-4b69-471f-89fe-aca516a015ab")
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
        X_eval = X_eval.append(pd.DataFrame({col:[i],"BSS":[bss]}))
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()
    plt.figure()
    X_eval.BSS.plot.hist( bins=10, alpha=0.5)
    plt.show()
    return(X_eval)

@transform_pandas(
    Output(rid="ri.vector.main.execute.4a661298-c3a3-434e-ae3d-47b3d3981c12"),
    final_model_1b_test_preds=Input(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea")
)
def eval_race_test(final_model_1b_test_preds):
    test_predictions = final_model_1b_test_preds
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
        print(f"brier: {bsl:0.4f}")
        print(f"BSS: {bss:0.4f}")
        print(f"rocauc: {roc:0.3f}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot()
        X_eval = X_eval.append(pd.DataFrame({col:[i],"BSS":[bss]}))
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()
    plt.figure()
    X_eval.BSS.plot.hist( bins=10, alpha=0.5)
    plt.show()
    return(X_eval)
    

@transform_pandas(
    Output(rid="ri.vector.main.execute.37e64286-c328-4988-aa99-0378c82328c8"),
    cv_pred=Input(rid="ri.foundry.main.dataset.e6b6df1b-4b69-471f-89fe-aca516a015ab")
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
    Output(rid="ri.vector.main.execute.a02d4734-2fb2-409e-b4ba-4b9ad238ecfe"),
    final_model_1b_test_preds=Input(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea")
)
def eval_sex_test(final_model_1b_test_preds):
    test_predictions = final_model_1b_test_preds
    from sklearn.metrics import brier_score_loss, roc_auc_score, RocCurveDisplay, auc, roc_curve
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    col = 'sex'
    print(test_predictions[col].unique())

    for i in ['MALE','FEMALE']:
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
        print(f"brier: {bsl:0.4f}")
        print(f"BSS: {bss:0.4f}")
        print(f"rocauc: {roc:0.3f}")
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot()
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.7806d367-76fd-4ce0-81a8-2f5093397749"),
    final_model_1b_test_preds=Input(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea")
)
def eval_test(final_model_1b_test_preds):

    from sklearn.metrics import brier_score_loss, roc_auc_score, auc, roc_curve
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    from sklearn.calibration import CalibrationDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    print(np.__version__)
    print(pd.__version__)

    true =  final_model_1b_test_preds['composite_outcome']
    pred = final_model_1b_test_preds['pred']
    
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
    Output(rid="ri.vector.main.execute.bad68378-a366-41a8-b3f2-0f6c7304d7c4"),
    cv_pred=Input(rid="ri.foundry.main.dataset.e6b6df1b-4b69-471f-89fe-aca516a015ab")
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
    Output(rid="ri.vector.main.execute.aa3b62fe-b74d-4e57-b814-465858b8e636"),
    final_model_1b_test_preds=Input(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea")
)
def eval_year_test(final_model_1b_test_preds):
    test_predictions = final_model_1b_test_preds
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
        print(f"brier: {bsl:0.4f}")
        print(f"BSS: {bss:0.4f}")
        print(f"rocauc: {roc:0.3f}")
        
        fpr, tpr, thresholds = roc_curve(true,  pred)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='')
        display.plot()
        #p = RocCurveDisplay.from_predictions(train_test_split['hosp'], train_test_split['pred'])
    plt.show()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.608c4b0b-116e-46dc-9b09-de2495ecddc6"),
    final_model_1b_test_preds=Input(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea"),
    fit_final_model_1b=Input(rid="ri.foundry.main.dataset.e5d4903f-64b0-438f-a9b1-c8d257cb89df")
)
def final_model_1b_shap(fit_final_model_1b, final_model_1b_test_preds):
    import lightgbm as lgb
    import pandas as pd
    import shap
    
    pred_cols = [x for x in final_model_1b_test_preds.columns if not x in ['composite_outcome', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                             'data_partner_id','pred']]
    X_test = final_model_1b_test_preds[pred_cols]
        
    model_string = str(fit_final_model_1b.iloc[0,0])

    bst = lgb.Booster(model_str=model_string)
    bst.params['objective'] = 'binary'

    feature_imp = pd.DataFrame({'Value':bst.feature_importance(),'Feature':X_test.columns})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False)

    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_test)
        
    for feat in feature_imp.Feature[0:25]:
        shap.dependence_plot(feat, shap_values[1], X_test, xmin = "percentile(1)", xmax = "percentile(99)", alpha = 0.4)

    ddf = pd.DataFrame(shap_values[1])
    ddf.columns = [f"{x}_shap" for x in X_test.columns]
    for x in X_test.columns:
        ddf[x] = X_test[x].values
    
    ddf['age'] = ddf['age'].astype('int32')

    return(ddf)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea"),
    features_test=Input(rid="ri.foundry.main.dataset.8fd7941f-6d5d-4d18-98a0-0a1a21071f48"),
    fit_final_model_1b=Input(rid="ri.foundry.main.dataset.e5d4903f-64b0-438f-a9b1-c8d257cb89df")
)
def final_model_1b_test_preds(fit_final_model_1b, features_test):
    import lightgbm as lgb
    import pandas as pd
    #print(lgb.__version__)

    pred_cols = [x for x in features_test.columns if not x in ['composite_outcome', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                             'data_partner_id']]
    X_test = features_test[pred_cols]
    model_string = str(fit_final_model_1b.iloc[0,0])

    bst = lgb.Booster(model_str=model_string)
    ypred = bst.predict(X_test)

    features_test['pred'] = ypred
    return features_test

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e5d4903f-64b0-438f-a9b1-c8d257cb89df"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
def fit_final_model_1b(features):

    import lightgbm as lgb
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    X_return = pd.DataFrame()
    #features = features_test

    pred_cols = [x for x in features.columns if not x in ['composite_outcome', 'person_id', 'cv_group','years_since_pandemic','sex','race','covid_year',
                                                                             'data_partner_id']]
    X_train = features[pred_cols]
    y_train = features['composite_outcome']

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

