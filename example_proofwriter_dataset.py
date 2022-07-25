import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from torchtextlogic.data_collators.bertnot_collator import (
    BERTNOTTextualEntailmentCollator,
)
from torchtextlogic.data_collators.proofwriter_collator import (
    ProofWriterProofGenerationAllCollator,
)
from torchtextlogic.data_collators.prover_collator import PRoverProofWriterCollator
from torchtextlogic.data_collators.ruletaker_collator import (
    RuleTakerProofWriterCollator,
)
from torchtextlogic.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset
from torchtextlogic.datasets.te.mnli_dataset import MNLIDataset
from torchtextlogic.datasets.te.rte_dataset import RTEDataset
from torchtextlogic.datasets.te.snli_dataset import SNLIDataset
from torchtextlogic.pl_models.bertnot import PLBERTNOT
from torchtextlogic.pl_models.proofwriter import PLProofWriter
from torchtextlogic.pl_models.prover import PLPRover
from torchtextlogic.pl_models.ruletaker import PLRuleTaker

MODEL = "bertnot"
DEVICE = "cpu"

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

    proofwriter_collator = ProofWriterProofGenerationAllCollator("t5-large")

    train_dataloader = DataLoader(train_dataset, 8, collate_fn=proofwriter_collator)
    val_dataloader = DataLoader(val_dataset, 8, collate_fn=proofwriter_collator)

    pl_proofwriter = PLProofWriter("t5-large", learning_rate=1e-5, weight_decay=0.1)

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

elif MODEL == "ruletaker":
    train_dataset = ProofWriterDataset("depth-5", "train", "proof_generation_all")
    val_dataset = ProofWriterDataset("depth-5", "val", "proof_generation_all")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="models/",
        filename="best_ruletaker-{epoch:02d}-{val_loss:.2f}",
    )

    ruletaker_collator = RuleTakerProofWriterCollator()

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
        filename="pretrained_bertnot.ckpt",
    )

    pl_bertnot = PLBERTNOT(
        "bert-base-cased", task="mlm", learning_rate=1e-5, batch_size=32, weight_decay=0
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
    val_dataset = SNLIDataset("val")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="models/",
        filename="snli_bertnot.ckpt",
    )

    train_dataloader = DataLoader(train_dataset, 32, collate_fn=bertnot_collator)
    val_dataloader = DataLoader(val_dataset, 32, collate_fn=bertnot_collator)

    pl_bertnot = PLBERTNOT.load_from_checkpoint(
        "models/pretrained_bertnot.ckpt",
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

    trainer.fit(pl_bertnot, train_dataloader, val_dataloader)

    ##############################################

    train_dataset = MNLIDataset("train")
    val_dataset = MNLIDataset("val")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="models/",
        filename="mnli_bertnot.ckpt",
    )

    train_dataloader = DataLoader(train_dataset, 32, collate_fn=bertnot_collator)
    val_dataloader = DataLoader(val_dataset, 32, collate_fn=bertnot_collator)

    pl_bertnot = PLBERTNOT.load_from_checkpoint(
        "models/pretrained_bertnot.ckpt",
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

    trainer.fit(pl_bertnot, train_dataloader, val_dataloader)

    ##############################################

    train_dataset = RTEDataset("train")
    val_dataset = RTEDataset("val")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="models/",
        filename="rte_bertnot.ckpt",
    )

    train_dataloader = DataLoader(train_dataset, 32, collate_fn=bertnot_collator)
    val_dataloader = DataLoader(val_dataset, 32, collate_fn=bertnot_collator)

    pl_bertnot = PLBERTNOT.load_from_checkpoint(
        "models/pretrained_bertnot.ckpt",
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

    trainer.fit(pl_bertnot, train_dataloader, val_dataloader)
