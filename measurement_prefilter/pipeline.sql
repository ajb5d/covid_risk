

@transform_pandas(
    Output(rid="ri.vector.main.execute.effe651f-6531-45a6-a011-16494fc1ab1f"),
    measurement_raw=Input(rid="ri.foundry.main.dataset.d6054221-ee0c-4858-97de-22292458fa19")
)
SELECT  measurement_id,
        CASE WHEN harmonized_unit_concept_id is NULL 
          THEN
            CASE WHEN harmonized_value_as_number is NULL THEN 'both null' ELSE 'unit null' END
          ELSE
            CASE WHEN harmonized_value_as_number is NULL THEN 'value null' ELSE 'both present' END
        END as mtype
FROM measurement_raw

@transform_pandas(
    Output(rid="ri.vector.main.execute.bbb6c356-df53-4611-92e7-98b8deefa378"),
    measurements=Input(rid="ri.foundry.main.dataset.389dc5dc-a75f-4570-9b25-6953794839f2")
)
SELECT variable, count(*)/1e6 as N
FROM measurements
GROUP BY variable
ORDER BY variable

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.389dc5dc-a75f-4570-9b25-6953794839f2"),
    Measurement_concepts=Input(rid="ri.foundry.main.dataset.a201c1a4-00c3-4ce7-8f5d-344f3ef49ceb"),
    measurement_raw=Input(rid="ri.foundry.main.dataset.d6054221-ee0c-4858-97de-22292458fa19")
)
SELECT measurement_raw.*,
       Measurement_concepts.variable
FROM measurement_raw
INNER JOIN Measurement_concepts
  ON measurement_raw.measurement_concept_id = Measurement_concepts.concept_id
WHERE measurement_raw.harmonized_value_as_number is not NULL
  

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1dd578c9-593f-4f90-ba55-1ad87d821be1"),
    Measurement_concepts=Input(rid="ri.foundry.main.dataset.a201c1a4-00c3-4ce7-8f5d-344f3ef49ceb"),
    observation_raw=Input(rid="ri.foundry.main.dataset.b998b475-b229-471c-800e-9421491409f3")
)
/* Check to see if any of our variables show up in the observations table.  Note that observations don't have a harmonized value. */
SELECT observation_raw.*,
       Measurement_concepts.variable
FROM observation_raw
INNER JOIN Measurement_concepts
  ON observation_raw.observation_concept_id = Measurement_concepts.concept_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.6407dfc7-3d0f-4053-a76d-3b30264dec25"),
    measurements=Input(rid="ri.foundry.main.dataset.389dc5dc-a75f-4570-9b25-6953794839f2")
)
SELECT *
FROM measurements
where variable = 'temp'
limit 1000

@transform_pandas(
    Output(rid="ri.vector.main.execute.c088a161-40fd-4fb1-bee5-32168110980d"),
    harmonized_value_count=Input(rid="ri.vector.main.execute.effe651f-6531-45a6-a011-16494fc1ab1f")
)
SELECT mtype, count(*) as N
FROM harmonized_value_count
GROUP BY mtype
ORDER BY N DESC

