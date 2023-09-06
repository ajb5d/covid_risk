

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8c97fd21-405f-423d-a1d5-bb5fa25eaa3b"),
    covid_cohort_raw=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    person_visits_18mo=Input(rid="ri.vector.main.execute.9c45f7c3-ad92-4549-abe6-b3d534b80664"),
    sdoh_imputed=Input(rid="ri.foundry.main.dataset.c7ad61f8-e802-4293-852d-84e77ec2ed2f")
)
SELECT covid_cohort_raw.*,
    DATEDIFF(covid_cohort_raw.first_COVID_ED_only_start_date, covid_cohort_raw.COVID_first_poslab_or_diagnosis_date) as EDlag,
    DATEDIFF(covid_cohort_raw.first_COVID_hospitalization_start_date, covid_cohort_raw.COVID_first_poslab_or_diagnosis_date) as hosplag,
    DATEDIFF(covid_cohort_raw.COVID_first_poslab_or_diagnosis_date, '2020-01-01') % 10 as cv_group,
    sdoh_imputed.poverty_pct,
    sdoh_imputed.poverty_pct_imputed_flag,
    sdoh_imputed.race_white_pct,
    sdoh_imputed.race_white_pct_imputed_flag,
    sdoh_imputed.edu_no_hs_pct,
    sdoh_imputed.edu_no_hs_pct_imputed_flag,
    sdoh_imputed.health_insurance_pct,
    sdoh_imputed.health_insurance_pct_imputed_flag,
    sdoh_imputed.md_per_thous_pop,
    sdoh_imputed.md_per_thous_pop_imputed_flag,
    sdoh_imputed.sdoh2,
    sdoh_imputed.sdoh2_imputed_flag
FROM covid_cohort_raw
INNER JOIN person_visits_18mo
   ON covid_cohort_raw.person_id = person_visits_18mo.person_id
LEFT JOIN sdoh_imputed
    ON covid_cohort_raw.person_id = sdoh_imputed.person_id
WHERE ((COVID_first_poslab_or_diagnosis_date > '2020-02-01' AND shift_date_yn = 'N')
         OR (COVID_first_poslab_or_diagnosis_date > '2020-01-01' AND shift_date_yn = 'Y')) 

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c2c6f85c-349a-4bb7-97a0-6d4bb9919547"),
    covid_cohort_raw=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198")
)
SELECT DATEDIFF(first_COVID_ED_only_start_date, COVID_first_poslab_or_diagnosis_date) as EDlag,
       DATEDIFF(first_COVID_hospitalization_start_date, COVID_first_poslab_or_diagnosis_date) as hosplag
FROM covid_cohort_raw
WHERE first_COVID_ED_only_start_date is not NULL or first_COVID_hospitalization_start_date is not NULL

@transform_pandas(
    Output(rid="ri.vector.main.execute.0e26a12b-aea0-449e-bdd4-f9c657a57b73"),
    sdoh_vars=Input(rid="ri.foundry.main.dataset.0ba44b7c-2167-4845-a647-97decc41e41c")
)
/*
 * I'd have preferred to use the median for these imputed values, but spark SQL doesn't have a median
 * operator, so we'll try the mean.  
*/
SELECT data_partner_id,
       avg(poverty_status) AS poverty_pct_avg,
       avg(White) AS race_white_pct_avg,
       avg(100 - (education_hs_diploma + education_associate_degree + education_bachelors_degree + education_graduate_or_professional_degree)) AS edu_no_hs_pct_avg,
       avg(health_insurance) AS health_insurance_pct_avg,
       avg(MDs_by_preferred_county) AS md_per_thous_pop_avg,
       avg(sdoh2_by_preferred_county) AS sdoh2_avg
FROM sdoh_vars
GROUP BY data_partner_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.c0cbf7d9-35ec-4411-87a7-fbe533f6c851"),
    partner_median_sdoh=Input(rid="ri.foundry.main.dataset.e62c62a5-d751-4de1-b2f4-562710094beb")
)
SELECT count(poverty_pct_med)/count(*) as avail_poverty,
    count(race_white_pct_med)/count(*) as avail_race,
    count(edu_no_hs_pct_med)/count(*) as avail_edu,
    count(health_insurance_med)/count(*) as avail_insurance,
    count(md_per_thous_pop_med)/count(*) as avail_md,
    count(sdoh2_med)/count(*) as avail_sdoh2
FROM partner_median_sdoh

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e62c62a5-d751-4de1-b2f4-562710094beb"),
    sdoh_vars=Input(rid="ri.foundry.main.dataset.0ba44b7c-2167-4845-a647-97decc41e41c")
)
SELECT data_partner_id,
       percentile_approx(poverty_status, 0.5) AS poverty_pct_med,
       percentile_approx(White, 0.5) AS race_white_pct_med,
       percentile_approx(100 - (education_hs_diploma + education_associate_degree + education_bachelors_degree + education_graduate_or_professional_degree), 0.5) AS edu_no_hs_pct_med,
       percentile_approx(health_insurance, 0.5) AS health_insurance_med,
       percentile_approx(MDs_by_preferred_county, 0.5) AS md_per_thous_pop_med,
       percentile_approx(sdoh2_by_preferred_county, 0.5) AS sdoh2_med
FROM sdoh_vars
GROUP BY data_partner_id
HAVING poverty_pct_med is not NULL

@transform_pandas(
    Output(rid="ri.vector.main.execute.9c45f7c3-ad92-4549-abe6-b3d534b80664"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6"),
    covid_cohort_raw=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
/* Find all the persons that had at least one visit, of any type, within the 18 months preceding their COVID diagnosis */
/* NOTE: spark SQL doesn't support a unit specification for datediff, so we have to work in days.  Also, apparently it has the arguments 
reversed from the way it's done in most other SQL dialects (i.e., datediff(end, start)) */
SELECT DISTINCT covid_cohort_raw.person_id
FROM visit_occurrence
INNER JOIN covid_cohort_raw
  ON visit_occurrence.person_id = covid_cohort_raw.person_id
INNER JOIN concept_set_members
  ON visit_occurrence.visit_concept_id = concept_set_members.concept_id
WHERE DATEDIFF(covid_cohort_raw.COVID_first_poslab_or_diagnosis_date, visit_occurrence.visit_start_date) <= 540 -- 540 days = 18 months.
  AND concept_set_members.codeset_id = 630441257   -- "clinical visit" v.2
  

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a77e54ab-ee34-4b0b-a13c-538c86830cb1"),
    covid_cohort=Input(rid="ri.foundry.main.dataset.8c97fd21-405f-423d-a1d5-bb5fa25eaa3b"),
    sdoh_selected_vars=Input(rid="ri.vector.main.execute.e7e14196-579d-4978-a6fc-031e96ef21c7")
)
SELECT
    data_partner_id
    , COUNT(*) AS total_c
    , 1.0 * SUM(CASE WHEN poverty_pct IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS miss_pct
FROM covid_cohort
GROUP BY data_partner_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c7ad61f8-e802-4293-852d-84e77ec2ed2f"),
    partner_median_sdoh=Input(rid="ri.foundry.main.dataset.e62c62a5-d751-4de1-b2f4-562710094beb"),
    sdoh_selected_vars=Input(rid="ri.vector.main.execute.e7e14196-579d-4978-a6fc-031e96ef21c7")
)
SELECT person_id,
       CASE WHEN poverty_pct IS NULL THEN poverty_pct_med ELSE poverty_pct END as poverty_pct,
       CASE WHEN poverty_pct IS NULL THEN 1 ELSE 0 END as poverty_pct_imputed_flag,
       CASE WHEN race_white_pct IS NULL THEN race_white_pct_med ELSE race_white_pct END as race_white_pct,
       CASE WHEN race_white_pct IS NULL THEN 1 ELSE 0 END as race_white_pct_imputed_flag,
       CASE WHEN edu_no_hs_pct IS NULL THEN edu_no_hs_pct_med ELSE edu_no_hs_pct END as edu_no_hs_pct,
       CASE WHEN edu_no_hs_pct IS NULL THEN 1 ELSE 0 END as edu_no_hs_pct_imputed_flag,
       CASE WHEN health_insurance_pct IS NULL THEN health_insurance_med ELSE health_insurance_pct END as health_insurance_pct,
       CASE WHEN health_insurance_pct IS NULL THEN 1 ELSE 0 END as health_insurance_pct_imputed_flag,
       CASE WHEN md_per_thous_pop IS NULL THEN md_per_thous_pop_med ELSE md_per_thous_pop END AS md_per_thous_pop,
       CASE WHEN md_per_thous_pop IS NULL THEN 1 ELSE 0 END as md_per_thous_pop_imputed_flag,
       CASE WHEN sdoh2 IS NULL THEN sdoh2_med ELSE sdoh2 END AS sdoh2,
       CASE WHEN sdoh2 IS NULL THEN 1 ELSE 0 END as sdoh2_imputed_flag
FROM sdoh_selected_vars
INNER JOIN partner_median_sdoh
    ON sdoh_selected_vars.data_partner_id = partner_median_sdoh.data_partner_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.e7e14196-579d-4978-a6fc-031e96ef21c7"),
    sdoh_vars=Input(rid="ri.foundry.main.dataset.0ba44b7c-2167-4845-a647-97decc41e41c")
)
SELECT data_partner_id,
       person_id,
       poverty_status as poverty_pct,
       CAST(White as FLOAT) AS race_white_pct,
       100 - (education_hs_diploma + education_associate_degree + education_bachelors_degree + education_graduate_or_professional_degree) AS edu_no_hs_pct,
       health_insurance AS health_insurance_pct,
       MDs_by_preferred_county AS md_per_thous_pop,
       sdoh2_by_preferred_county AS sdoh2
FROM sdoh_vars
where poverty_status is not NULL

@transform_pandas(
    Output(rid="ri.vector.main.execute.a4617983-dd50-4955-85ef-a006bd451efd"),
    concept=Input(rid="ri.foundry.main.dataset.5cb3c4a3-327a-47bf-a8bf-daf0cafe6772"),
    visit_occurrence=Input(rid="ri.foundry.main.dataset.911d0bb2-c56e-46bd-af4f-8d9611183bb7")
)
SELECT DISTINCT concept.concept_class_id
FROM concept
INNER JOIN visit_occurrence
  ON visit_occurrence.visit_concept_id = concept.concept_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.48fa8f5f-b812-4ae4-ad08-312b50d3ba5d"),
    covid_cohort_raw=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198")
)
SELECT COVID_associated_ED_only_visit_indicator as ED_only, COVID_associated_hospitalization_indicator as hosp, count(*)/1e6 as N_million
FROM covid_cohort_raw
GROUP BY COVID_associated_ED_only_visit_indicator, COVID_associated_hospitalization_indicator
ORDER BY N_million DESC

@transform_pandas(
    Output(rid="ri.vector.main.execute.70c4d48f-52d0-4a9d-a7b2-9a0dd81a42cf"),
    lag_dx_to_admit=Input(rid="ri.foundry.main.dataset.c2c6f85c-349a-4bb7-97a0-6d4bb9919547")
)
SELECT count(*) as N_zero_lag
FROM lag_dx_to_admit
WHERE hosplag <= 0

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2e86b468-db06-48b0-8176-25994bae4c6a"),
    concept=Input(rid="ri.foundry.main.dataset.5cb3c4a3-327a-47bf-a8bf-daf0cafe6772")
)
SELECT *
FROM concept
where domain_id LIKE '%Visit%'

