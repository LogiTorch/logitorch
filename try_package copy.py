import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from data_collators.ruletaker_collator import RuleTakerProofWriterCollator
from pl_models.proofwriter import PLProofWriter
from pl_models.prover import PLPRover
from pl_models.ruletaker import PLRuleTaker
from torchtextlogic.data_collators.proofwriter_collator import (
    ProofWriterProofGenerationAllCollator,
)
from torchtextlogic.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset
from torchtextlogic.models.proofwriter import ProofWriter

dataset = ProofWriterDataset("depth-5", "train", "proof_generation_all")

collator_fn = ProofWriterProofGenerationAllCollator("t5-base")
train_dataloader = DataLoader(dataset, 20, collate_fn=collator_fn)
model = PLProofWriter("t5-base")
# cnt = 0


# model = PRoverTrainer("roberta-base")
trainer = pl.Trainer(accelerator="cpu")
trainer.fit(model, train_dataloader)
