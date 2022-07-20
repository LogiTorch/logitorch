import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from torchtextlogic.data_collators.proofwriter_collator import (
    ProofWriterProofGenerationAllCollator,
)
from torchtextlogic.data_collators.prover_collator import PRoverProofWriterCollator
from torchtextlogic.data_collators.ruletaker_collator import (
    RuleTakerProofWriterCollator,
)
from torchtextlogic.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset
from torchtextlogic.pl_models.proofwriter import PLProofWriter
from torchtextlogic.pl_models.prover import PLPRover
from torchtextlogic.pl_models.ruletaker import PLRuleTaker

MODEL = "ruletaker"
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

    train_dataloader = DataLoader(train_dataset, 32, collate_fn=proofwriter_collator)
    val_dataloader = DataLoader(val_dataset, 32, collate_fn=proofwriter_collator)

    pl_proofwriter = PLProofWriter("t5-large")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback], accelerator=DEVICE, max_epochs=10
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

    train_dataloader = DataLoader(train_dataset, 32, collate_fn=prover_collator)
    val_dataloader = DataLoader(val_dataset, 32, collate_fn=prover_collator)

    pl_prover = PLPRover("roberta-large")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback], accelerator=DEVICE, max_epochs=10
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

    ruletaker_collator = RuleTakerProofWriterCollator("roberta-base")

    train_dataloader = DataLoader(train_dataset, 32, collate_fn=ruletaker_collator)
    val_dataloader = DataLoader(val_dataset, 32, collate_fn=ruletaker_collator)

    pl_ruletaker = PLRuleTaker("roberta-base")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback], accelerator=DEVICE, max_epochs=10
    )

    trainer.fit(pl_ruletaker, train_dataloader, val_dataloader)
