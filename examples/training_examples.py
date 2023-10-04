import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from logitorch.data_collators.bertnot_collator import BERTNOTTextualEntailmentCollator
from logitorch.data_collators.proofwriter_collator import (
    ProofWriterProofGenerationAllCollator,
)
from logitorch.data_collators.prover_collator import PRoverProofWriterCollator
from logitorch.data_collators.fld_collator import FLDProofGenerationAllCollator
from logitorch.data_collators.ruletaker_collator import (
    RuleTakerCollator,
    RuleTakerProofWriterCollator,
)
from logitorch.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset
from logitorch.datasets.proof_qa.fld_dataset import FLDDataset
from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
from logitorch.datasets.te.mnli_dataset import MNLIDataset
from logitorch.datasets.te.rte_dataset import RTEDataset
from logitorch.datasets.te.snli_dataset import SNLIDataset
from logitorch.pl_models.bertnot import PLBERTNOT
from logitorch.pl_models.proofwriter import PLProofWriter
from logitorch.pl_models.fld import PLFLDAllAtOnceProver
from logitorch.pl_models.prover import PLPRover
from logitorch.pl_models.ruletaker import PLRuleTaker

# MODEL = "proofwriter"
# DEVICE = "cpu"

MODEL = "FLD"
DEVICE = "gpu"


def main():
    if MODEL == "proofwriter":
        train_dataset = ProofWriterDataset("depth-5", "train", "proof_generation_all")
        val_dataset = ProofWriterDataset("depth-5", "val", "proof_generation_all")

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath="models/",
            filename="best_proofwriter-{epoch:02d}-{val_loss:.2f}",
        )

        proofwriter_collator = ProofWriterProofGenerationAllCollator(
            "google/t5-v1_1-large"
        )

        train_dataloader = DataLoader(train_dataset, 8, collate_fn=proofwriter_collator)
        val_dataloader = DataLoader(val_dataset, 8, collate_fn=proofwriter_collator)

        pl_proofwriter = PLProofWriter(
            "google/t5-v1_1-large", learning_rate=1e-5, weight_decay=0.1
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=5,
        )

        trainer.fit(pl_proofwriter, train_dataloader, val_dataloader)

    elif MODEL == "prover":
        train_dataset = ProofWriterDataset("depth-5", "train", "proof_generation_all")
        val_dataset = ProofWriterDataset("depth-5", "val", "proof_generation_all")

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath="models/",
            filename="best_prover-{epoch:02d}-{val_loss:.2f}",
        )

        prover_collator = PRoverProofWriterCollator("roberta-large")

        train_dataloader = DataLoader(train_dataset, 16, collate_fn=prover_collator)
        val_dataloader = DataLoader(val_dataset, 16, collate_fn=prover_collator)

        pl_prover = PLPRover("roberta-large", learning_rate=1e-5, weight_decay=0.1)

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=10,
        )

        trainer.fit(pl_prover, train_dataloader, val_dataloader)

    elif MODEL == "FLD":
        train_dataset = FLDDataset("FLD.v2", "train", "proof_generation_all")
        val_dataset = FLDDataset("FLD.v2", "val", "proof_generation_all", max_samples=100)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath="models/",
            filename="best_fld-{epoch:02d}-{val_loss:.2f}",
            # every_n_train_steps=1000,  # every_n_train_steps requires monitor="train_loss"
        )

        fld_collator = FLDProofGenerationAllCollator(
            "t5-base", log_examples=False,
        )

        train_dataloader = DataLoader(train_dataset, 4, collate_fn=fld_collator)
        val_dataloader = DataLoader(val_dataset, 4, collate_fn=fld_collator)

        pl_proofwriter = PLFLDAllAtOnceProver(
            "t5-base", learning_rate=1e-4, weight_decay=0.1, warmup_steps=1000,
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=False,
            accelerator=DEVICE,
            accumulate_grad_batches=16,
            max_epochs=40,
            # max_steps=100,
            # max_steps=20000,
        )

        trainer.fit(pl_proofwriter, train_dataloader, val_dataloader)

    elif MODEL == "ruletaker":
        train_dataset = RuleTakerDataset("depth-5", "train")
        val_dataset = RuleTakerDataset("depth-5", "val")

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath="models/",
            filename="best_ruletaker_ruletaker-{epoch:02d}-{val_loss:.2f}",
        )

        ruletaker_collator = RuleTakerCollator()

        train_dataloader = DataLoader(train_dataset, 16, collate_fn=ruletaker_collator)
        val_dataloader = DataLoader(val_dataset, 16, collate_fn=ruletaker_collator)

        pl_ruletaker = PLRuleTaker(weight_decay=0.1, learning_rate=1e-5)

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=10,
        )

        trainer.fit(pl_ruletaker, train_dataloader, val_dataloader)


    elif MODEL == "bertnot":

        checkpoint_callback = ModelCheckpoint(
            monitor=None,
            save_top_k=1,
            dirpath="models/",
            filename="pretrained_bertnot",
        )

        pl_bertnot = PLBERTNOT(
            "bert-base-cased",
            task="mlm",
            learning_rate=1e-5,
            batch_size=32,
            weight_decay=0,
            num_labels=3,
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=5,
        )

        trainer.fit(pl_bertnot)

        ##############################################

        bertnot_collator = BERTNOTTextualEntailmentCollator("bert-base-cased")

        train_dataset = SNLIDataset("train")

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            dirpath="models/",
            filename="snli_bertnot",
        )

        train_dataloader = DataLoader(train_dataset, 32, collate_fn=bertnot_collator)

        pl_bertnot = PLBERTNOT.load_from_checkpoint(
            "models/pretrained_bertnot.ckpt",
            pretrained_model="bert-base-cased",
            task="te",
            learning_rate=1e-5,
            weight_decay=0.1,
            num_labels=3,
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=3,
        )

        trainer.fit(pl_bertnot, train_dataloader)

        ##############################################

        train_dataset = MNLIDataset("train")

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            dirpath="models/",
            filename="mnli_bertnot",
        )

        train_dataloader = DataLoader(train_dataset, 32, collate_fn=bertnot_collator)

        pl_bertnot = PLBERTNOT.load_from_checkpoint(
            "models/pretrained_bertnot.ckpt",
            pretrained_model="bert-base-cased",
            task="te",
            learning_rate=2e-5,
            weight_decay=0.0,
            num_labels=3,
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=3,
        )

        trainer.fit(pl_bertnot, train_dataloader)

        ##############################################

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            dirpath="models/",
            filename="pretrained_bertnot_2",
        )

        pl_bertnot = PLBERTNOT(
            "bert-base-cased",
            task="mlm",
            learning_rate=1e-5,
            batch_size=32,
            weight_decay=0,
            num_labels=2,
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=5,
        )

        trainer.fit(pl_bertnot)

        ##############################################

        train_dataset = RTEDataset("train")

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            dirpath="models/",
            filename="rte_bertnot",
        )

        train_dataloader = DataLoader(train_dataset, 32, collate_fn=bertnot_collator)

        pl_bertnot = PLBERTNOT.load_from_checkpoint(
            "models/pretrained_bertnot_2.ckpt",
            pretrained_model="bert-base-cased",
            task="te",
            learning_rate=2e-5,
            weight_decay=0.0,
            num_labels=2,
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=50,
        )

        trainer.fit(pl_bertnot, train_dataloader)
    elif MODEL == "rte":
        train_dataset = RTEDataset("train")

        bertnot_collator = BERTNOTTextualEntailmentCollator("bert-base-cased")

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            dirpath="models/",
            filename="rte_bertnot",
        )

        train_dataloader = DataLoader(train_dataset, 32, collate_fn=bertnot_collator)

        pl_bertnot = PLBERTNOT.load_from_checkpoint(
            "models/pretrained_bertnot_2.ckpt",
            pretrained_model="bert-base-cased",
            task="te",
            learning_rate=2e-5,
            weight_decay=0.0,
            num_labels=2,
        )

        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            auto_lr_find=True,
            accelerator=DEVICE,
            max_epochs=50,
        )

        trainer.fit(pl_bertnot, train_dataloader)


if __name__ == "__main__":
    main()
