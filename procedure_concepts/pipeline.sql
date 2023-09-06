

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.96c4d476-b6e8-489d-9a8b-8631f5b2274e"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
/* Pull the concept_ids for dialysis */ 

SELECT
    concept_id
   ,concept_set_name
   ,'dialysis' as proc_type
FROM concept_set_members
WHERE 
    codeset_id = 777835196 
    AND is_most_recent_version = TRUE
UNION ALL
/* Pull the concept_ids for NIPPV */ 

SELECT
    concept_id
   ,concept_set_name
   ,'nippv' as proc_type
FROM concept_set_members
WHERE 
    codeset_id = 23680970 
    AND is_most_recent_version = TRUE

