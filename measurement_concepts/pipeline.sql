

@transform_pandas(
    Output(rid="ri.vector.main.execute.f561e1c4-d719-4930-873a-1ab5a83fdaba"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
/* Verify that each codeset_id refers to a single version */
SELECT codeset_id, count(distinct version) as N_version
FROM concept_set_members
GROUP BY codeset_id
ORDER BY N_version DESC

@transform_pandas(
    Output(rid="ri.vector.main.execute.dfbf9318-6f54-4728-a6e8-4305944fd10f"),
    measurement_concepts=Input(rid="ri.foundry.main.dataset.a201c1a4-00c3-4ce7-8f5d-344f3ef49ceb")
)
SELECT concept_id, count(*) as N 
FROM measurement_concepts
GROUP BY concept_id
ORDER BY N DESC
--LIMIT 25

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a201c1a4-00c3-4ce7-8f5d-344f3ef49ceb"),
    concept_set_members=Input(rid="ri.foundry.main.dataset.e670c5ad-42ca-46a2-ae55-e917e3e161b6")
)
/* Sysbp: this concept set seems reasonably mature; it is marked as "finished" and has two reviews */
SELECT *, 'sysbp' as variable
FROM concept_set_members
WHERE codeset_id = 186465804 AND version = 1

UNION ALL

/* Diabp: same remarks as sysbp above */
SELECT *, 'diabp' as variable
FROM concept_set_members
WHERE codeset_id = 573275931 AND version = 1

UNION ALL

/* heart rate: This one is marked as "finished", but has not been reviewed, though it has been marked as "candidate for review".
I looked through the included concepts, and they all seem reasonable, but it's hard to spot any that might be missing without a 
comprehensive knowledge of the concept set hierarchy.  All told, this one seems a little less official than the two blood pressure sets,
 but it's way ahead of any of the other contenders, all of which are marked "under construction".   

Should we switch to using is_most_recent_version instead of expicitly selecting the version?
*/
SELECT *, 'heartrate' as variable
FROM concept_set_members
WHERE codeset_id = 596956209 AND is_most_recent_version=true
UNION ALL

/* respiratory rate: marked as "finished", has two versions, and has two reviews 
*/
SELECT *, 'resprate' as variable
FROM concept_set_members
WHERE codeset_id = 286601963 AND is_most_recent_version=true
UNION ALL

/* temperature: marked as "finished" and has 3 reviews 
*/
SELECT *, 'temp' as variable
FROM concept_set_members
WHERE codeset_id = 656562966 AND is_most_recent_version=true
UNION ALL

/* BMI: marked as "finished", has two versions, and has two reviews 
*/
SELECT *, 'bmi' as variable
FROM concept_set_members
WHERE codeset_id = 65622096 AND is_most_recent_version=true
UNION ALL

/* SpO2: marked as "finished", has 4 versions, but no reviews
*/
SELECT *, 'spo2' as variable
FROM concept_set_members
WHERE codeset_id = 780678652 AND is_most_recent_version=true
UNION ALL

/* A1C: marked as "finished", has 2 versions, and 2 reviews
*/
SELECT *, 'a1c' as variable
FROM concept_set_members
WHERE codeset_id = 381434987 AND is_most_recent_version=true
UNION ALL

/* white count: marked as "finished", has 2 versions, and 2 reviews
*/
SELECT *, 'wbc' as variable
FROM concept_set_members
WHERE codeset_id = 138719030 AND is_most_recent_version=true
UNION ALL

/* glucose: marked as "finished", has 2 reviews
*/
SELECT *, 'glucose' as variable
FROM concept_set_members
WHERE codeset_id = 59698832 AND is_most_recent_version=true
UNION ALL

/* sodium: marked as "finished", has 1 review
*/
SELECT *, 'sodium' as variable
FROM concept_set_members
WHERE codeset_id = 887473517 AND is_most_recent_version=true
UNION ALL

/* potassium: marked as "finished", has 2 reviews
*/
SELECT *, 'potassium' as variable
FROM concept_set_members
WHERE codeset_id = 622316047 AND is_most_recent_version=true
UNION ALL

/* chloride: marked as "finished", has 2 reviews
*/
SELECT *, 'chloride' as variable
FROM concept_set_members
WHERE codeset_id = 733538531 AND is_most_recent_version=true
UNION ALL

/* albumin: marked as "finished", has 1 review
Only one concept ID so maybe a bit suspect, though it does include descendants
*/
SELECT *, 'albumin' as variable
FROM concept_set_members
WHERE codeset_id = 104464584 AND is_most_recent_version=true
UNION ALL

/* ALT: marked as "finished", has 2 reviews
*/
SELECT *, 'alt' as variable
FROM concept_set_members
WHERE codeset_id = 538737057 AND is_most_recent_version=true
UNION ALL

/* AST: marked as "finished", has 2 reviews
*/
SELECT *, 'ast' as variable
FROM concept_set_members
WHERE codeset_id = 248480621 AND is_most_recent_version=true
UNION ALL

/* bilirubin: marked as "finished", has 1 review
*/
SELECT *, 'bilirubin' as variable
FROM concept_set_members
WHERE codeset_id = 586434833 AND is_most_recent_version=true
UNION ALL

/* BUN: marked as "finished", has 2 reviews
*/
SELECT *, 'bun' as variable
FROM concept_set_members
WHERE codeset_id = 139231433 AND is_most_recent_version=true
UNION ALL

/* creatinine: marked as "finished", has 2 versions, and one review
*/
SELECT *, 'creatinine' as variable
FROM concept_set_members
WHERE codeset_id = 615348047 AND is_most_recent_version=true
UNION ALL

/* hemoglobin: marked as "finished", has 2 reviews
*/
SELECT *, 'hemoglobin' as variable
FROM concept_set_members
WHERE codeset_id = 28177118 AND is_most_recent_version=true
UNION ALL

/* platelets: marked as "finished", has 2 versions, and 2 reviews
*/
SELECT *, 'platelets' as variable
FROM concept_set_members
WHERE codeset_id = 167697906 AND is_most_recent_version=true
UNION ALL

/* ferritin: used by a bunch of projects
*/

SELECT *, 'ferritin' AS variable
FROM concept_set_members
WHERE codeset_id = 317388455 AND is_most_recent_version=true
UNION ALL

/* d-dimer: used by core n3c project
*/

SELECT *, 'ddimer' AS variable
FROM concept_set_members
WHERE codeset_id = 442059787 AND is_most_recent_version=true

