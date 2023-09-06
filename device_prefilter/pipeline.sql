

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.46febf69-6067-4084-9231-cafe68abdf77"),
    Device_concepts=Input(rid="ri.foundry.main.dataset.7a155b61-f44e-427e-a42e-896700d6fda6"),
    device_exposure=Input(rid="ri.foundry.main.dataset.d685db48-6583-43d6-8dc5-a9ebae1a827a")
)
SELECT device_exposure.*
FROM device_exposure
INNER JOIN Device_concepts
  ON device_exposure.device_concept_id = Device_concepts.concept_id

  

