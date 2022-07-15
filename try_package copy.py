import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

from models.prover import PRover
from torchtextlogic.data_collators.prover_collator import PRoverProofWriterCollator
from torchtextlogic.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset
from trainers.prover_trainer import PRoverTrainer

dataset = ProofWriterDataset("depth-2", "val", "proof_generation_all")

collator_fn = PRoverProofWriterCollator("roberta-base")
train_dataloader = DataLoader(dataset, 20, collate_fn=collator_fn)
model = PRover("roberta-base")
cnt = 0
for i in dataset:
    print(i)
    print(model.predict(i[0], i[1], i[2]))
    cnt += 1
    if cnt == 88:
        break

# model = PRoverTrainer("roberta-base")
# trainer = pl.Trainer(accelerator="cpu")
# trainer.fit(model, train_dataloader)
