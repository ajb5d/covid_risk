

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.620faefb-6673-40d3-86cb-f68750da4711"),
    Devices=Input(rid="ri.foundry.main.dataset.46febf69-6067-4084-9231-cafe68abdf77"),
    covid_cohort=Input(rid="ri.foundry.main.dataset.8c97fd21-405f-423d-a1d5-bb5fa25eaa3b")
)

SELECT Devices.device_exposure_id,
       Devices.person_id,
       Devices.device_concept_name,
       Devices.device_exposure_start_date,
       DATEDIFF(Devices.device_exposure_start_date, covid_cohort.COVID_first_poslab_or_diagnosis_date) AS days_since_dx,
       DATEDIFF(Devices.device_exposure_start_date, covid_cohort.first_COVID_hospitalization_start_date) AS days_since_hosp
FROM Devices
INNER JOIN covid_cohort
  ON covid_cohort.person_id = Devices.person_id
WHERE Devices.device_exposure_start_date between covid_cohort.first_COVID_hospitalization_start_date and covid_cohort.first_COVID_hospitalization_end_date

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.0f267c6d-02b5-4b60-ab96-64dfcba79ff4"),
    covid_cohort=Input(rid="ri.foundry.main.dataset.8c97fd21-405f-423d-a1d5-bb5fa25eaa3b"),
    drugs=Input(rid="ri.foundry.main.dataset.85d8bc3c-dac3-4316-992f-0e84fcd8d7d6")
)

SELECT drugs.drug_exposure_id,
       drugs.person_id,
       drugs.drug_concept_name,
       drugs.drug_exposure_start_date,
       drugs.drug_exposure_end_date,
       DATEDIFF(drugs.drug_exposure_start_date, covid_cohort.COVID_first_poslab_or_diagnosis_date) AS days_since_dx,
       DATEDIFF(drugs.drug_exposure_start_date, covid_cohort.first_COVID_hospitalization_start_date) AS days_since_hosp
FROM drugs
INNER JOIN covid_cohort
  ON covid_cohort.person_id = drugs.person_id
WHERE drugs.drug_exposure_start_date between covid_cohort.first_COVID_hospitalization_start_date and covid_cohort.first_COVID_hospitalization_end_date

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.09d4f5cb-6cf3-481e-a5ef-9840b7824c64"),
    covid_cohort=Input(rid="ri.foundry.main.dataset.8c97fd21-405f-423d-a1d5-bb5fa25eaa3b"),
    measurements=Input(rid="ri.foundry.main.dataset.389dc5dc-a75f-4570-9b25-6953794839f2")
)

SELECT measurements.measurement_id,
       measurements.person_id,
       measurements.variable,
       measurements.measurement_date,  -- A lot of the measurement times are missing, so I don't think the datetime is going to be very useful
       measurements.operator_concept_id,
       measurements.value_as_number,
       measurements.unit_concept_id,
       measurements.harmonized_unit_concept_id,
       measurements.harmonized_value_as_number,
       DATEDIFF(measurements.measurement_date, covid_cohort.COVID_first_poslab_or_diagnosis_date) AS days_since_dx,
       DATEDIFF(measurements.measurement_date, covid_cohort.first_COVID_hospitalization_start_date) AS days_since_hosp
FROM measurements
INNER JOIN covid_cohort
  ON covid_cohort.person_id = measurements.person_id
WHERE DATEDIFF(measurements.measurement_date, covid_cohort.COVID_first_poslab_or_diagnosis_date) >= -180
  AND DATEDIFF(measurements.measurement_date, covid_cohort.COVID_first_poslab_or_diagnosis_date) <= 60 -- The LL template uses a 60-day window to define covid-related outcomes, so any measurement after this can't possibly be predictive of anything we're interested in.

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.788d3d68-59c8-4364-a436-224d61266b13"),
    covid_cohort=Input(rid="ri.foundry.main.dataset.8c97fd21-405f-423d-a1d5-bb5fa25eaa3b"),
    procedures=Input(rid="ri.foundry.main.dataset.df79e9e6-1e11-461d-814f-e7e17eb718e2")
)

SELECT procedures.procedure_occurrence_id,
       procedures.person_id,
       procedures.procedure_concept_name,
       procedures.proc_type,
       procedures.procedure_date,
       DATEDIFF(procedures.procedure_date, covid_cohort.COVID_first_poslab_or_diagnosis_date) AS days_since_dx,
       DATEDIFF(procedures.procedure_date, covid_cohort.first_COVID_hospitalization_start_date) AS days_since_hosp
FROM procedures
INNER JOIN covid_cohort
  ON covid_cohort.person_id = procedures.person_id
WHERE procedures.procedure_date between covid_cohort.first_COVID_hospitalization_start_date and covid_cohort.first_COVID_hospitalization_end_date

@transform_pandas(
    Output(rid="ri.vector.main.execute.9675ce62-2c6d-40d9-a0c1-55dd6ff775af"),
    covid_cohort_measurements=Input(rid="ri.foundry.main.dataset.09d4f5cb-6cf3-481e-a5ef-9840b7824c64")
)
/* For each patient and variable, find the last measurement on or before the diagnosis measurement_date
*/

SELECT *
FROM covid_cohort_measurements

@transform_pandas(
    Output(rid="ri.vector.main.execute.eb6bfe10-f004-4e10-9e25-0ae730eebef4"),
    covid_cohort_measurements=Input(rid="ri.foundry.main.dataset.09d4f5cb-6cf3-481e-a5ef-9840b7824c64")
)
SELECT min(days_since_dx) as mindays_dx,
        max(days_since_dx) as maxdays_dx,
        min(days_since_hosp) as mindays_hosp,
        max(days_since_hosp) as maxdays_hosp
FROM covid_cohort_measurements

