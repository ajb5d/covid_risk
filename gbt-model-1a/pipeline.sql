

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    Model_features_1a_train_imputed=Input(rid="ri.foundry.main.dataset.6c3860bf-2ed6-45e6-9ae8-2724b79cb599")
)
SELECT f.person_id
      ,cv_group
      ,ll.data_partner_id
      ,age
      ,CAST(DATEDIFF(ll.COVID_first_poslab_or_diagnosis_date, '2020-01-01')*1.0/365 as float) as years_since_pandemic
      ,date_part('YEAR',ll.COVID_first_poslab_or_diagnosis_date) as covid_year
      --,ll.state
    ,f.sex
    ,f.race
	,CAST(nvax AS INT) as nvax
	,poverty_pct / 100 as poverty_frac
	,race_white_pct / 100 as race_white_frac
	,edu_no_hs_pct / 100 as edu_no_hs_frac
	,health_insurance_pct / 100 as health_insurance_frac
	,md_per_thous_pop / 10 as md_per_hundred_pop
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
      ,hosp
FROM Model_features_1a_train_imputed as f
LEFT JOIN Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_ as ll on ll.person_id=f.person_id
WHERE f.sex IS NOT NULL and data_partner_id IS NOT NULL and data_partner_id NOT IN (578,861,798,266,399) and f.race<>'Hispanic or Latino'
--4,949,672

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.64ef334a-cb91-459f-9dec-cf646d71676c"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    Model_features_1a_train=Input(rid="ri.foundry.main.dataset.873101b7-6339-48bd-ae98-dcfa21de9efa")
)
SELECT f.person_id
      ,cv_group
      ,ll.data_partner_id
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
      ,tuberculosis
      --,liver_mld
      ,liver_mod
      --,thalassemia
      --,rheum
      ,dementia
      ,chf
      --,substance
      --,downsynd
      ,kidney
      ,maligcanc
      ,diabcx
      --,diabun
      ,cerebvasc
      ,periphvasc
      ,pregnancy
      ,hf
      --,paralysis
      --,psychosis
      ,obesity
      ,coronary
      ,csteroid
      --,depression
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
      ,hosp
FROM Model_features_1a_train as f
LEFT JOIN Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_ as ll on ll.person_id=f.person_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e46f083a-ec97-4db9-bdcd-9e344d8528dc"),
    Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_=Input(rid="ri.foundry.main.dataset.75d7da57-7b0e-462c-b41d-c9ef4f756198"),
    Model_features_1a_test_imputed=Input(rid="ri.foundry.main.dataset.06583938-2bdb-41d2-b8bf-d6335f1252eb")
)
SELECT f.person_id
      ,cv_group
      ,ll.data_partner_id
      ,age
      ,CAST(DATEDIFF(ll.COVID_first_poslab_or_diagnosis_date, '2020-01-01')*1.0/365 as float) as years_since_pandemic
      ,date_part('YEAR',ll.COVID_first_poslab_or_diagnosis_date) as covid_year
      --,ll.state
    ,f.sex
    ,f.race
	,CAST(nvax AS INT) as nvax
	,poverty_pct / 100 as poverty_frac
	,race_white_pct / 100 as race_white_frac
	,edu_no_hs_pct / 100 as edu_no_hs_frac
	,health_insurance_pct / 100 as health_insurance_frac
	,md_per_thous_pop / 10 as md_per_hundred_pop
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
      ,hosp
FROM Model_features_1a_test_imputed as f
LEFT JOIN Logic_Liaison_Covid_19_Patient_Summary_Facts_Table_LDS_ as ll on ll.person_id=f.person_id
WHERE f.sex IS NOT NULL and data_partner_id IS NOT NULL and data_partner_id NOT IN (578,861,798,266,399) and f.race<>'Hispanic or Latino'
--4,949,672

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9905a39f-086a-44eb-85da-5a534bcde1a9"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
SELECT data_partner_id
      ,avg(hosp) as hosp_outcome
      ,count(*) as patients
      ,sum(hosp) as hospitalizations
FROM features
GROUP BY data_partner_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a0d1f83c-c0f8-484c-b1e2-6fb59371325b"),
    features=Input(rid="ri.foundry.main.dataset.70469268-15f7-49d4-8bc7-5247d6483035")
)
SELECT race, count(*) as cnt, sum(hosp) as hosp
FROM features
GROUP BY race
ORDER BY cnt DESC

