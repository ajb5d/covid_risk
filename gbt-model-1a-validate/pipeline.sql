

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9a62489c-1dc4-4b8a-83e0-0ef643af8e9a"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
SELECT data_partner_id
      ,count(*) as cnt
      ,avg(hosp) as hosp
FROM features
GROUP BY data_partner_id
ORDER BY cnt

@transform_pandas(
    Output(rid="ri.vector.main.execute.6df46815-5ee5-40a5-9564-60d788574293"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
SELECT cv_group
      ,count(*) as cnt
FROM features
GROUP BY cv_group
ORDER BY cnt

@transform_pandas(
    Output(rid="ri.vector.main.execute.a3a90b9d-9cd3-4343-8fad-b7f01c642160"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
SELECT *
FROM features
WHERE sex IS NULL

