

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cd7f3f06-cee9-4a17-974a-40a39f2dd46a"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
SELECT data_partner_id
      ,count(*) as cnt
      ,avg(hosp) as hosp
FROM features
GROUP BY data_partner_id
ORDER BY cnt

@transform_pandas(
    Output(rid="ri.vector.main.execute.d45d05e9-0901-442d-9a36-3b1c2f09f0f9"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
SELECT cv_group
      ,count(*) as cnt
FROM features
GROUP BY cv_group
ORDER BY cnt

@transform_pandas(
    Output(rid="ri.vector.main.execute.903cc2ca-5be6-4c32-90f8-4edc369b852e"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
SELECT *
FROM features
WHERE sex IS NULL

