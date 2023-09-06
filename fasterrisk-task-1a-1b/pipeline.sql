

@transform_pandas(
    Output(rid="ri.vector.main.execute.59876e57-990e-43a6-b8b7-3e28ce8c2d43"),
    fr_task_1a=Input(rid="ri.foundry.main.dataset.7192d603-eef1-417d-906c-cc9a99fbdd3a")
)
SELECT parameter, count(*) as N, avg(coefficient) as meancoef
FROM fr_task_1a
WHERE parmid > 1
GROUP BY parameter
ORDER BY N DESC

@transform_pandas(
    Output(rid="ri.vector.main.execute.6edb44ab-0e30-42bf-85cb-324d464852ab"),
    fr_task_1b_all_vars=Input(rid="ri.foundry.main.dataset.bc5f0217-4c58-4a7f-8155-fd438be6af7e")
)
SELECT parameter, count(*) as N, avg(coefficient) as meancoef
FROM fr_task_1b_all_vars
WHERE parmid > 1
GROUP BY parameter
ORDER BY N DESC

@transform_pandas(
    Output(rid="ri.vector.main.execute.d3214507-003d-497a-84df-87dd5d6e8662"),
    fr_task_1b_no_fm_vars=Input(rid="ri.foundry.main.dataset.d95f1c2c-8c38-4045-9036-3e9d80487a16")
)
SELECT parameter, count(*) as N, avg(coefficient) as meancoef
FROM fr_task_1b_no_fm_vars
WHERE parmid > 1
GROUP BY parameter
ORDER BY N DESC

@transform_pandas(
    Output(rid="ri.vector.main.execute.3c9085d3-5893-46ed-88b9-581222e25ef6"),
    Model_features_1a=Input(rid="ri.foundry.main.dataset.2edf8c28-e744-4efa-8414-4625a2397010")
)
SELECT  case when age < 40 then 1 else 0 end as AgeLess40,
        case when age >= 40 and age <= 60 then 1 else 0 end as Age40to60,
        case when age > 60 then 1 else 0 end as AgeGreater60, 
        case when sex = 'FEMALE' then 1 else 0 end as SexFemale, 
        case when poverty_pct <= 10 then 1 else 0 end as PovertyLess10,
        case when poverty_pct > 10 and poverty_pct <= 25 then 1 else 0 end as Poverty10to25,
        case when poverty_pct > 25 and poverty_pct <= 50 then 1 else 0 end as Poverty25t050,
        case when poverty_pct > 50 then 1 else 0 end as PovertyGreater50,
        -- outcome
        case when hosp == 1 then 1 else -1 end as hosp
FROM Model_features_1a
LIMIT 100000

@transform_pandas(
    Output(rid="ri.vector.main.execute.e16624d9-44cd-41a8-b257-e3a3390ef3e6"),
    Model_features_1a_train_imputed=Input(rid="ri.foundry.main.dataset.6c3860bf-2ed6-45e6-9ae8-2724b79cb599")
)

SELECT  CAST(age AS INT) as age, 
        case when sex = 'FEMALE' then 1 else 0 end as SexFemale, 
	-- CAST(ROUND(bmi, 0) AS INT) as bmi,
	-- comorbidities
	tuberculosis,
	liver_mld,
	liver_mod,
	thalassemia,
	rheum,
	dementia,
	chf,
	substance,
	downsynd,
	kidney,
	maligcanc,
	diabcx,
	diabun,
	cerebvasc,
	periphvasc,
	pregnancy,
	hf,
	paralysis,
	psychosis,
	obesity,
	coronary,
	csteroid,
	depression,
	metacanc,
	hiv,
	chroniclung,
	peptic_ulcer,
	sicklecell,
	mi,
	cardiomyopathy,
	htn,
	immunodef,
	pulm_emb,
	tobacco,
	transplant,
	CASE WHEN nvax > 0 THEN 1 ELSE 0 END as isvaxed,

	-- sdoh vars
    CAST(ROUND(poverty_pct, 0) AS INT) as poverty_pct,
	CAST(ROUND(race_white_pct, 0) AS INT) as race_white_pct,
	CAST(ROUND(edu_no_hs_pct, 0) AS INT) as edu_no_hs_pct,
	CAST(ROUND(health_insurance_pct, 0) AS INT) as health_insurance_pct,
	CAST(ROUND(md_per_thous_pop, 0) AS INT) as md_per_thous_pop,
        -- outcome
    case when hosp == 1 then 1 else -1 end as hosp
FROM Model_features_1a_train_imputed
WHERE age >= 18 AND cv_group = 7
ORDER BY person_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2ea7125a-95d8-4123-868c-ec3136efee48"),
    Model_features_1b_train_imputed=Input(rid="ri.foundry.main.dataset.7fe29c67-b5ba-475d-85ce-9db8d8228616")
)

SELECT  CAST(age AS INT) as age, 
    	case when sex = 'FEMALE' then 1 else 0 end as SexFemale, 
	CAST(ROUND(bmi, 0) AS INT) as bmi, -- note, frequently missing
	-- comorbidities
	tuberculosis,
	liver_mld,
	liver_mod,
	thalassemia,
	rheum,
	dementia,
	chf,
	substance,
	downsynd,
	kidney,
	maligcanc,
	diabcx,
	diabun,
	cerebvasc,
	periphvasc,
	pregnancy,
	hf,
	paralysis,
	psychosis,
	obesity,
	coronary,
	csteroid,
	depression,
	metacanc,
	hiv,
	chroniclung,
	peptic_ulcer,
	sicklecell,
	mi,
	cardiomyopathy,
	htn,
	immunodef,
	pulm_emb,
	tobacco,
	transplant,
	CASE WHEN nvax > 0 THEN 1 ELSE 0 END as isvaxed,

	-- test results
	antibody_pos,
	antibody_neg,

	-- sdoh vars
    CAST(ROUND(poverty_pct, 0) AS INT) as poverty_pct,
	CAST(ROUND(race_white_pct, 0) AS INT) as race_white_pct,
	CAST(ROUND(edu_no_hs_pct, 0) AS INT) as edu_no_hs_pct,
	CAST(ROUND(health_insurance_pct, 0) AS INT) as health_insurance_pct,
	CAST(ROUND(md_per_thous_pop, 0) AS INT) as md_per_thous_pop,

    -- vitals and labs included only if no more than 70% are missing.
    -- vitals: all of these are frequently misisng
    CAST(ROUND(sysbp, 0) AS INT) as sysbp,
    CAST(ROUND(diabp, 0) AS INT) as diabp,
    CAST(ROUND(heartrate, 0) AS INT) as heartrate,
    CAST(ROUND(resprate, 0) AS INT) as resprate,
    ROUND(temp, 1) as temp,
    CAST(ROUND(spo2, 0) AS INT) as spo2,

    -- labs: a1c excluded because it's missing 82% of the time.
    -- Others are included, but they are frequently missing (15-30%)
    CAST(ROUND(wbc, 0) AS INT) as wbc,
    CAST(ROUND(glucose, 0) AS INT) as glucose,
    CAST(ROUND(sodium, 0) AS INT) as sodium,
    ROUND(potassium, 1) as potassium,
    CAST(ROUND(chloride, 0) AS INT) as chloride,
    ROUND(albumin, 1) as albumin,
    CAST(ROUND(alt, 0) AS INT) as alt,
    CAST(ROUND(ast, 0) AS INT) as ast,
    ROUND(bilirubin, 1) as bilirubin,
    CAST(ROUND(bun, 0) AS INT) as bun,
    ROUND(creatinine, 1) as creatinine,
    ROUND(hemoglobin, 1) as hemoglobin,
    CAST(ROUND(platelets, 0) AS INT) as platelets,

    -- treatments
    treat_steroid,
    remdisivir,

    -- outcome: death w/in 60 days, or ventilation (invasive or non),
    -- ecmo, vasopressors, or dialysis
    CASE WHEN death60 = 1 OR imv = 1 OR nippv = 1 OR ecmo = 1
    	 OR vasopressor = 1 or dialysis = 1 THEN 1 ELSE -1 END AS deathetc

FROM Model_features_1b_train_imputed
WHERE age >= 18 AND cv_group % 2 = 0
ORDER BY person_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.a42ee3a8-0ccb-468e-bce0-457c73d7d12f"),
    Model_features_1a_test_imputed=Input(rid="ri.foundry.main.dataset.06583938-2bdb-41d2-b8bf-d6335f1252eb")
)
SELECT
    CASE WHEN age <= 34 THEN -2 ELSE 0 END AS age34
    , CASE WHEN age <= 50 THEN -3 ELSE 0 END as age51
    , CASE WHEN age <= 76 THEN -2 ELSE 0 END AS age76
    , CASE WHEN kidney > 0 THEN 2 ELSE 0 END AS kidney
    , CASE WHEN diabun > 0 THEN 2 ELSE 0 END AS diabun
    , CASE WHEN pregnancy > 0 THEN 4 ELSE 0 END AS pregnancy
    , CASE WHEN csteroid > 0 THEN 1 ELSE 0 END AS csteroid
    , CASE WHEN htn > 0 THEN 1 ELSE 0 END AS htn
    , CASE WHEN nvax > 0 THEN -4 ELSE 0 END AS isvaxed
    , CASE WHEN edu_no_hs_pct <= 23 THEN -2 ELSE 0 END AS edu_no_hs_pct
    , hosp
FROM Model_features_1a_test_imputed

@transform_pandas(
    Output(rid="ri.vector.main.execute.b6dbf4e4-31c8-466a-9a63-8e0c65a68a99"),
    processed_feat_1a=Input(rid="ri.vector.main.execute.e16624d9-44cd-41a8-b257-e3a3390ef3e6")
)
SELECT count(*)/1e3
FROM processed_feat_1a

@transform_pandas(
    Output(rid="ri.vector.main.execute.5f698396-2dfa-4d83-8c6e-1b2bd6d8c149"),
    processed_features_1b=Input(rid="ri.foundry.main.dataset.2ea7125a-95d8-4123-868c-ec3136efee48")
)
SELECT count(*)/1000
FROM processed_features_1b

