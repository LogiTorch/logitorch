import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from logitorch.data_collators.proofwriter_collator import (
    ProofWriterProofGenerationAllCollator,
)
from logitorch.data_collators.prover_collator import PRoverProofWriterCollator
from logitorch.data_collators.fld_collator import FLDProofGenerationAllCollator
from logitorch.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset
from logitorch.datasets.proof_qa.fld_dataset import FLDDataset
from logitorch.pipelines.exceptions import ModelNotCompatibleError
from logitorch.pl_models.proofwriter import PLProofWriter
from logitorch.pl_models.prover import PLPRover
from logitorch.pl_models.fld import FLDAllAtOnceProver

PROOFWRITER_COMPATIBLE_MODELS = (PLProofWriter, PLPRover)
FLD_COMPATIBLE_MODELS = (FLDAllAtOnceProver,)


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
    """
    Executes the proofwriter pipeline for training a proof generation model.

    Args:
        model (nn.Module): The proof generation model.
        dataset_name (str): The name of the dataset.
        task (str, optional): The task to perform. Defaults to "proof_generation_all".
        open_world_assumption (bool, optional): Whether to use open world assumption. Defaults to False.
        saved_model_path (str, optional): The path to save the trained model. Defaults to "/".
        saved_model_name (str, optional): The name of the saved model. Defaults to "best_model".
        batch_size (int, optional): The batch size for training. Defaults to 1.
        epochs (int, optional): The number of training epochs. Defaults to 1.
        accelerator (str, optional): The accelerator to use (e.g., "cpu", "gpu"). Defaults to "cpu".
        gpus (int, optional): The number of GPUs to use. Defaults to 0.

    Raises:
        ModelNotCompatibleError: If the provided model is not compatible with the proofwriter pipeline.

    """
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


def fld_pipeline(
    model: nn.Module,
    dataset_name: str,
    task: str = "proof_generation_all",
    saved_model_path: str = "/",
    saved_model_name: str = "best_model",
    batch_size: int = 4,
    accum_steps: int = 16,
    epochs: int = 40,
    accelerator: str = "cpu",
    gpus: int = 0,
):
    """
    Executes the fld pipeline for training a proof generation model.

    Args:
        model (nn.Module): The proof generation model.
        dataset_name (str): The name of the dataset.
        task (str, optional): The task to perform. Defaults to "proof_generation_all".
        saved_model_path (str, optional): The path to save the trained model. Defaults to "/".
        saved_model_name (str, optional): The name of the saved model. Defaults to "best_model".
        batch_size (int, optional): The batch size for training. Defaults to 4.
        accum_steps (int, optional): The number of accumulation steps. Defaults to 16.
        epochs (int, optional): The number of training epochs. Defaults to 40.
        accelerator (str, optional): The accelerator to use (e.g., "cpu", "gpu"). Defaults to "cpu".
        gpus (int, optional): The number of GPUs to use. Defaults to 0.

    Raises:
        ModelNotCompatibleError: If the provided model is not compatible with the fld pipeline.

    """
    try:
        if isinstance(model, FLD_COMPATIBLE_MODELS):
            train_dataset = FLDDataset(
                dataset_name, "train", task,
            )
            val_dataset = FLDDataset(
                dataset_name, "val", task, max_samples=100,
            )

            if isinstance(model, FLDAllAtOnceProver):
                fld_collate_fn = FLDProofGenerationAllCollator(
                    "t5-base", log_examples=False,
                )
            else:
                raise ValueError()

            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, collate_fn=fld_collate_fn
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, collate_fn=fld_collate_fn
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
                auto_lr_find=False,
                accelerator=accelerator,
                accumulate_grad_batches=accum_steps,
                max_epochs=epochs,
                gpus=gpus,
            )

            trainer.fit(model, train_dataloader, val_dataloader)
        else:
            raise ModelNotCompatibleError(FLD_COMPATIBLE_MODELS)
    except ModelNotCompatibleError as err:
        print(err.message)
