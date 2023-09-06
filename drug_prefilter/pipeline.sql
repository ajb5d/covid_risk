

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.85d8bc3c-dac3-4316-992f-0e84fcd8d7d6"),
    drug_concepts=Input(rid="ri.foundry.main.dataset.e65b2c65-b9a4-43a8-8ae1-942dd786b410"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.ec252b05-8f82-4f7f-a227-b3bb9bc578ef")
)
SELECT drug_exposure.*
FROM drug_exposure
INNER JOIN drug_concepts on drug_exposure.drug_concept_id=drug_concepts.concept_id

