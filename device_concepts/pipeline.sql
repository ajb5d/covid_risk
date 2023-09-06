

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7a155b61-f44e-427e-a42e-896700d6fda6"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
/* Pull the concept_ids for nippv devices */ 

SELECT
    concept_id
FROM concept_set_members
WHERE 
    codeset_id = 23680970 
    AND is_most_recent_version = TRUE

