

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3bdca309-692e-4215-9f30-61d0fdc2e651"),
    Final_model_1b_test_preds=Input(rid="ri.foundry.main.dataset.b1d0a7bf-3690-46db-9af1-1905807d19ea")
)
SELECT
    composite_outcome
    , pred
    , sex
    , race
FROM Final_model_1b_test_preds

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.87ab1dc2-c9e9-43f4-8de9-5f7c4a33d10e"),
    Final_model_1a_test_preds=Input(rid="ri.foundry.main.dataset.41000225-d0d5-4ba1-af56-3839588d2161")
)
SELECT
    hosp
    , pred
    , sex
    , race
FROM Final_model_1a_test_preds

