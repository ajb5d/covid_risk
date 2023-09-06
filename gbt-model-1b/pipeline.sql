

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.aba5f884-4e47-43ee-b46c-f0893d0870cc"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    Model_features_1b_train=Input(rid="ri.foundry.main.dataset.c2bf0d7f-0cdf-4945-819a-c95e310a39b3")
)
SELECT f.person_id
      ,cv_group
      ,DATEDIFF(f.hospenddate, f.hospdate) as LOS
      ,ll.data_partner_id
      ,CAST(DATEDIFF(ll.COVID_first_poslab_or_diagnosis_date, '2020-01-01')*1.0/365 as float) as years_since_pandemic
      ,date_part('YEAR',ll.COVID_first_poslab_or_diagnosis_date) as covid_year
      --,ll.state
    ,f.sex
    ,f.race
      ,bmi
      --,sex
      ,age
      ,age_over_89
      ,poverty_pct
      ,health_insurance_pct
      ,race_white_pct
      ,edu_no_hs_pct
      ,sysbp
      ,diabp
      ,heartrate
      ,resprate
      ,temp
      ,spo2
      ,a1c
      ,wbc
      ,glucose
      ,sodium
      ,potassium
      ,chloride
      ,albumin
      ,alt
      ,ast
      ,bilirubin
      ,bun
      ,creatinine
      ,hemoglobin
      ,platelets
      ,ferritin
      ,ddimer
      ,tuberculosis
      ,liver_mld
      ,liver_mod
      ,thalassemia
      ,rheum
      ,dementia
      ,chf
      ,substance
      ,downsynd
      ,kidney
      ,maligcanc
      ,diabcx
      ,diabun
      ,cerebvasc
      ,periphvasc
      ,pregnancy
      ,hf
      ,paralysis
      ,psychosis
      ,obesity
      ,coronary
      ,csteroid
      ,depression
      ,metacanc
      ,hiv
      ,chroniclung
      ,peptic_ulcer
      ,sicklecell
      ,mi
      ,cardiomyopathy
      ,htn
      ,immunodef
      ,pulm_emb
      ,tobacco
      ,transplant
      ,CASE WHEN ecmo=1 or death_hosp=1 or vasopressor=1 or dialysis=1 or nippv=1 THEN 1 ELSE 0 END as composite_outcome
      ,CASE WHEN nippv_days_since_hosp<=1 or dialysis_days_since_hosp<=1 or vasopressor_days_since_hosp<=1 or 
                (death_hosp=1 and DATEDIFF(f.hospenddate, f.hospdate)<=1) THEN 1 ELSE 0 END as outcome_filter
FROM Model_features_1b_train as f
LEFT JOIN Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_ as ll on ll.person_id=f.person_id
--WHERE nippv_days_since_hosp<=2 or dialysis_days_since_hosp<=2 or vasopressor_days_since_hosp<=2 or 
--(death_hosp=1 and DATEDIFF(f.hospenddate, f.hospdate)<=2) 
--4,949,672

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6"),
    Features_prep=Input(rid="ri.foundry.main.dataset.aba5f884-4e47-43ee-b46c-f0893d0870cc")
)
SELECT person_id
      ,cv_group
      --,LOS
      ,data_partner_id
      ,sex
      ,race
      ,years_since_pandemic
      ,covid_year
      ,bmi
      ,age
      ,age_over_89
      ,poverty_pct
      ,health_insurance_pct
      ,race_white_pct
      ,edu_no_hs_pct
      ,sysbp
      ,diabp
      --,heartrate
      ,resprate
      ,temp
      ,spo2
      ,a1c
      ,wbc
      ,glucose
      ,sodium
      ,potassium
      ,chloride
      ,albumin
      ,alt
      ,ast
      ,bilirubin
      ,bun
      ,creatinine
      ,hemoglobin
      ,platelets
      ,ferritin
      ,ddimer
      ,tuberculosis
      ,liver_mld
      ,liver_mod
      ,thalassemia
      ,rheum
      ,dementia
      ,chf
      ,substance
      ,downsynd
      ,kidney
      ,maligcanc
      ,diabcx
      ,diabun
      ,cerebvasc
      ,periphvasc
      ,pregnancy
      ,hf
      ,paralysis
      ,psychosis
      ,obesity
      ,coronary
      ,csteroid
      ,depression
      ,metacanc
      ,hiv
      ,chroniclung
      ,peptic_ulcer
      ,sicklecell
      ,mi
      ,cardiomyopathy
      ,htn
      ,immunodef
      ,pulm_emb
      ,tobacco
      ,transplant
      ,composite_outcome
FROM Features_prep
WHERE outcome_filter=0 and data_partner_id<>285 and race<>'Hispanic or Latino'

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.3f453b44-a920-45ce-8ea4-3997cadb97f0"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    Model_features_1b_test_imputed=Input(rid="ri.foundry.main.dataset.4a387a72-9d00-4959-98e4-d0b052754649")
)
SELECT f.person_id
      ,cv_group
      ,DATEDIFF(f.hospenddate, f.hospdate) as LOS
      ,ll.data_partner_id
      ,CAST(DATEDIFF(ll.COVID_first_poslab_or_diagnosis_date, '2020-01-01')*1.0/365 as float) as years_since_pandemic
      ,date_part('YEAR',ll.COVID_first_poslab_or_diagnosis_date) as covid_year
      --,ll.state
    ,f.sex
    ,f.race
      ,bmi
      --,sex
      ,age
      ,age_over_89
      ,poverty_pct
      ,health_insurance_pct
      ,race_white_pct
      ,edu_no_hs_pct
      ,sysbp
      ,diabp
      ,heartrate
      ,resprate
      ,temp
      ,spo2
      ,a1c
      ,wbc
      ,glucose
      ,sodium
      ,potassium
      ,chloride
      ,albumin
      ,alt
      ,ast
      ,bilirubin
      ,bun
      ,creatinine
      ,hemoglobin
      ,platelets
      ,ferritin
      ,ddimer
      ,tuberculosis
      ,liver_mld
      ,liver_mod
      ,thalassemia
      ,rheum
      ,dementia
      ,chf
      ,substance
      ,downsynd
      ,kidney
      ,maligcanc
      ,diabcx
      ,diabun
      ,cerebvasc
      ,periphvasc
      ,pregnancy
      ,hf
      ,paralysis
      ,psychosis
      ,obesity
      ,coronary
      ,csteroid
      ,depression
      ,metacanc
      ,hiv
      ,chroniclung
      ,peptic_ulcer
      ,sicklecell
      ,mi
      ,cardiomyopathy
      ,htn
      ,immunodef
      ,pulm_emb
      ,tobacco
      ,transplant
      ,CASE WHEN ecmo=1 or death_hosp=1 or vasopressor=1 or dialysis=1 or nippv=1 THEN 1 ELSE 0 END as composite_outcome
      ,CASE WHEN nippv_days_since_hosp<=1 or dialysis_days_since_hosp<=1 or vasopressor_days_since_hosp<=1 or 
                (death_hosp=1 and DATEDIFF(f.hospenddate, f.hospdate)<=1) THEN 1 ELSE 0 END as outcome_filter
FROM Model_features_1b_test_imputed as f
LEFT JOIN Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_ as ll on ll.person_id=f.person_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8fd7941f-6d5d-4d18-98a0-0a1a21071f48"),
    features_prep_test=Input(rid="ri.foundry.main.dataset.3f453b44-a920-45ce-8ea4-3997cadb97f0")
)
SELECT person_id
      ,cv_group
      --,LOS
      ,data_partner_id
      ,sex
      ,race
      ,years_since_pandemic
      ,covid_year
      ,bmi
      ,age
      ,age_over_89
      ,poverty_pct
      ,health_insurance_pct
      ,race_white_pct
      ,edu_no_hs_pct
      ,sysbp
      ,diabp
      --,heartrate
      ,resprate
      ,temp
      ,spo2
      ,a1c
      ,wbc
      ,glucose
      ,sodium
      ,potassium
      ,chloride
      ,albumin
      ,alt
      ,ast
      ,bilirubin
      ,bun
      ,creatinine
      ,hemoglobin
      ,platelets
      ,ferritin
      ,ddimer
      ,tuberculosis
      ,liver_mld
      ,liver_mod
      ,thalassemia
      ,rheum
      ,dementia
      ,chf
      ,substance
      ,downsynd
      ,kidney
      ,maligcanc
      ,diabcx
      ,diabun
      ,cerebvasc
      ,periphvasc
      ,pregnancy
      ,hf
      ,paralysis
      ,psychosis
      ,obesity
      ,coronary
      ,csteroid
      ,depression
      ,metacanc
      ,hiv
      ,chroniclung
      ,peptic_ulcer
      ,sicklecell
      ,mi
      ,cardiomyopathy
      ,htn
      ,immunodef
      ,pulm_emb
      ,tobacco
      ,transplant
      ,composite_outcome
FROM features_prep_test
WHERE outcome_filter=0 and data_partner_id<>285 and race<>'Hispanic or Latino'

@transform_pandas(
    Output(rid="ri.vector.main.execute.72e722a1-4423-4564-9e66-f30bee7d480e"),
    Features_prep=Input(rid="ri.foundry.main.dataset.aba5f884-4e47-43ee-b46c-f0893d0870cc")
)
SELECT *
FROM Features_prep
WHERE heartrate=0

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0bd2d053-d8b8-4836-b566-3476cb44deca"),
    Features_prep=Input(rid="ri.foundry.main.dataset.aba5f884-4e47-43ee-b46c-f0893d0870cc")
)
SELECT avg(composite_outcome) as outcome_rate
      ,sum(composite_outcome) as outcomes
      ,count(*) as total_rows
      ,sum(outcome_filter) as filtered_outcomes
      ,sum(outcome_filter)*1.0/sum(composite_outcome)  as filtered_outcome_rate
FROM Features_prep
--WHERE composite_outcome=1

@transform_pandas(
    Output(rid="ri.vector.main.execute.4fa9bc4c-8804-4236-ac36-7e5c19fc3c74"),
    Features_prep=Input(rid="ri.foundry.main.dataset.aba5f884-4e47-43ee-b46c-f0893d0870cc")
)
SELECT avg(composite_outcome)
FROM Features_prep
WHERE outcome_filter=0

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.339a14f6-5cc0-4ed3-ba9d-b99f5ddcc4cf"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
SELECT *
FROM features
WHERE heartrate=0

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.272c9445-b269-4c11-a60a-b6b382f5c134"),
    features=Input(rid="ri.foundry.main.dataset.0e2f9e94-c239-4ffa-86e0-ef88196398e6")
)
SELECT data_partner_id
      ,count(*) as cnt
      ,avg(composite_outcome) as composite_outcome
FROM features
GROUP BY data_partner_id
ORDER BY cnt DESC

