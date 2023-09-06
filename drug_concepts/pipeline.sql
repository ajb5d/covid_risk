

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e65b2c65-b9a4-43a8-8ae1-942dd786b410"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
/* Pull the concept_ids for vasopressor exposure */ 

SELECT
    concept_id
FROM concept_set_members
WHERE 
    codeset_id = 249190512 
    AND is_most_recent_version = TRUE

