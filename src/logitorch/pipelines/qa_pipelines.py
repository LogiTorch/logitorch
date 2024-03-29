import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from logitorch.data_collators.ruletaker_collator import RuleTakerCollator
from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
from logitorch.pipelines.exceptions import ModelNotCompatibleError
from logitorch.pl_models.ruletaker import PLRuleTaker

RULETAKER_COMPATIBLE_MODELS = PLRuleTaker


def ruletaker_pipeline(
    model: nn.Module,
    dataset_name: str,
    saved_model_path: str,
    saved_model_name: str,
    batch_size: int,
    epochs: int,
    accelerator: str = "cpu",
    gpus: int = 0,
):
    """
    Runs the RuleTaker pipeline for training a model.

    Args:
        model (nn.Module): The model to be trained.
        dataset_name (str): The name of the dataset.
        saved_model_path (str): The path to save the trained model.
        saved_model_name (str): The name of the saved model file.
        batch_size (int): The batch size for training.
        epochs (int): The number of training epochs.
        accelerator (str, optional): The accelerator to use for training. Defaults to "cpu".
        gpus (int, optional): The number of GPUs to use for training. Defaults to 0.

    Raises:
        ModelNotCompatibleError: If the model is not compatible with RuleTaker.

    """
    try:
        if isinstance(model, RULETAKER_COMPATIBLE_MODELS):
            if isinstance(model, PLRuleTaker):
                train_dataset = RuleTakerDataset(dataset_name, "train")
                val_dataset = RuleTakerDataset(dataset_name, "val")

                ruletaker_collate_fn = RuleTakerCollator()

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    collate_fn=ruletaker_collate_fn,
                )
                val_dataloader = DataLoader(
                    val_dataset, batch_size=batch_size, collate_fn=ruletaker_collate_fn
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
            raise ModelNotCompatibleError(RULETAKER_COMPATIBLE_MODELS)

    except ModelNotCompatibleError as err:
        print(err.message)
