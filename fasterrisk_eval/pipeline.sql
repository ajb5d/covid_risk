

@transform_pandas(
    Output(rid="ri.vector.main.execute.c8041687-96e5-4d2b-a072-0d69243a861f"),
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
WHERE age >= 18 AND cv_group = 9
ORDER BY person_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.c890f9af-d439-4d53-a40f-169d05e45b1c"),
    Model_features_1a_test_imputed=Input(rid="ri.foundry.main.dataset.06583938-2bdb-41d2-b8bf-d6335f1252eb")
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
FROM Model_features_1a_test_imputed
WHERE age >= 18
ORDER BY person_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.de52d46d-73f8-4dd7-964e-ac7e654d4c36"),
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
WHERE age >= 18 AND cv_group % 2 = 1
ORDER BY person_id

@transform_pandas(
    Output(rid="ri.vector.main.execute.ca322e99-581d-4af8-b6a4-314470dda1c1"),
    Model_features_1b_test_imputed=Input(rid="ri.foundry.main.dataset.4a387a72-9d00-4959-98e4-d0b052754649")
)
SELECT CAST(age AS INT) as age, 
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
FROM Model_features_1b_test_imputed
WHERE age >= 18
ORDER BY person_id

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.02eecf97-fab7-4cf6-9c89-9d0b9d0e4537"),
    fr_1b_test_set_eval=Input(rid="ri.foundry.main.dataset.95df0b48-eb91-48cd-9477-3855645c2bc4")
)
SELECT
    score
    , ROUND(model_prob,3) AS predicted_risk
    , N
    , ROUND(empirical_prob, 3) AS observed_rate
FROM fr_1b_test_set_eval
ORDER BY score ASC

