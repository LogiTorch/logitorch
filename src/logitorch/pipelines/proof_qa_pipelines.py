import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from logitorch.data_collators.proofwriter_collator import (
    ProofWriterProofGenerationAllCollator,
)
from logitorch.data_collators.prover_collator import PRoverProofWriterCollator
from logitorch.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset
from logitorch.pipelines.exceptions import ModelNotCompatibleError
from logitorch.pl_models.proofwriter import PLProofWriter
from logitorch.pl_models.prover import PLPRover

PROOFWRITER_COMPATIBLE_MODELS = (PLProofWriter, PLPRover)


def proofwriter_pipeline(
    model: nn.Module,
    dataset_name: str,
    task: str = "proof_generation_all",
    open_world_assumption: bool = False,
    saved_model_path: str = "/",
    saved_model_name: str = "best_model",
    batch_size: int = 1,
    epochs: int = 1,
    accelerator: str = "cpu",
    gpus: int = 0,
):
    try:
        if isinstance(model, PROOFWRITER_COMPATIBLE_MODELS):
            train_dataset = ProofWriterDataset(
                dataset_name, "train", task, open_world_assumption
            )
            val_dataset = ProofWriterDataset(
                dataset_name, "val", task, open_world_assumption
            )

            if isinstance(model, PLProofWriter):
                proofwriter_collate_fn = ProofWriterProofGenerationAllCollator(
                    model.pretrained_model
                )
            elif isinstance(model, PLPRover):
                proofwriter_collate_fn = PRoverProofWriterCollator(
                    model.pretrained_model
                )

            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, collate_fn=proofwriter_collate_fn
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, collate_fn=proofwriter_collate_fn
            )

            checkpoint_callback = ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                dirpath=saved_model_path,
                filename=saved_model_name,
            )

            trainer = pl.Trainer(
                callbacks=[checkpoint_callback],
                max_epochs=epochs,
                accelerator=accelerator,
                gpus=gpus,
            )

            trainer.fit(model, train_dataloader, val_dataloader)

        else:
            raise ModelNotCompatibleError(PROOFWRITER_COMPATIBLE_MODELS)
    except ModelNotCompatibleError as err:
        print(err.message)
