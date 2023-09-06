

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.df79e9e6-1e11-461d-814f-e7e17eb718e2"),
    procedure_concepts=Input(rid="ri.foundry.main.dataset.96c4d476-b6e8-489d-9a8b-8631f5b2274e"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.f6f0b5e0-a105-403a-a98f-0ee1c78137dc")
)
SELECT procedure_occurrence.*,
       procedure_concepts.proc_type
FROM procedure_occurrence
INNER JOIN procedure_concepts
  ON procedure_occurrence.procedure_concept_id = procedure_concepts.concept_id

  

