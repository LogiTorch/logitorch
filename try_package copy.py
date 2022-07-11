from torch.utils.data.dataloader import DataLoader

from models.prover import PRover
from torchtextlogic.data_collators.prover_collator import PRoverProofWriterCollator
from torchtextlogic.datasets.proof_qa.proofwriter_dataset import ProofWriterDataset

dataset = ProofWriterDataset("depth-2", "val", "proof_generation_all")

collator_fn = PRoverProofWriterCollator("roberta-base")
train_dataloader = DataLoader(dataset, 20, collate_fn=collator_fn)

model = PRover("roberta-base")
for i in train_dataloader:
    x = i[0]
    p = i[1]
    n = i[2]
    e = i[3]
    l = i[4]
    t = model(x, p, n, e, l)
    print(t)
    break
