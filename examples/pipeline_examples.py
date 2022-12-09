from logitorch.pipelines.qa_pipelines import ruletaker_pipeline
from logitorch.pl_models.ruletaker import PLRuleTaker

model = PLRuleTaker(learning_rate=1e-5, weight_decay=0.1)

ruletaker_pipeline(
    model=model,
    dataset_name="depth-5",
    saved_model_name="models/",
    saved_model_path="best_ruletaker",
    batch_size=32,
    epochs=10,
    accelerator="cpu",
)
