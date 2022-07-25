from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torchtextlogic.datasets.proof_qa.proofwriter_dataset import (
    PROOFWRITER_LABEL_TO_ID,
    ProofWriterDataset,
)
from torchtextlogic.pl_models.proofwriter import PLProofWriter
from torchtextlogic.pl_models.prover import PLPRover
from torchtextlogic.pl_models.ruletaker import PLRuleTaker

MODEL = "proofwriter"
DEVICE = "cuda:0"


def parse_facts_rules(facts, rules):
    sentences = []
    for s in facts.values():
        sentences.append(s)
    for s in rules.values():
        sentences.append(s)
    context = "".join(sentences)
    return context


proofwriter_test_datasets = ["depth-5", "birds-electricity"]

if MODEL == "proofwriter":
    model = PLProofWriter.load_from_checkpoint(
        "models/best_proofwriter-epoch=04-val_loss=0.02.ckpt",
        pretrained_model="t5-large",
    )
    model.to(DEVICE)
    model.eval()
    for d in proofwriter_test_datasets:
        test_dataset = ProofWriterDataset(d, "test", "proof_generation_all")
        with open(f"proofwriter_{d}.txt", "w") as out:
            y_preds = []
            y_trues = []
            for i in tqdm(test_dataset):
                context = parse_facts_rules(i[0], i[1])
                y_pred = model.predict(context, i[2], device=DEVICE)
                if "True" in y_pred:
                    y_pred = 1
                else:
                    y_pred = 0
                y_true = PROOFWRITER_LABEL_TO_ID[str(i[3])]
                y_preds.append(y_pred)
                y_trues.append(y_true)
            out.write(str(accuracy_score(y_trues, y_preds)))

elif MODEL == "prover":
    model = PLPRover.load_from_checkpoint(
        "models/best_prover-epoch=04-val_loss=0.10.ckpt",
        pretrained_model="roberta-large",
    )
    model.to(DEVICE)
    model.eval()
    for d in proofwriter_test_datasets:
        test_dataset = ProofWriterDataset(d, "test", "proof_generation_all")
        with open(f"prover_{d}.txt", "w") as out:
            y_preds = []
            y_trues = []
            for i in tqdm(test_dataset):
                y_pred = model.predict(i[0], i[1], i[2], device=DEVICE)
                y_true = PROOFWRITER_LABEL_TO_ID[str(i[3])]
                y_preds.append(y_pred)
                y_trues.append(y_true)
            out.write(str(accuracy_score(y_trues, y_preds)))

elif MODEL == "ruletaker":
    model = PLRuleTaker.load_from_checkpoint(
        "models/best_ruletaker-epoch=04-val_loss=0.03.ckpt"
    )
    model.to(DEVICE)
    model.eval()
    for d in proofwriter_test_datasets:
        test_dataset = ProofWriterDataset(d, "test", "proof_generation_all")
        with open(f"ruletaker_{d}.txt", "w") as out:
            y_preds = []
            y_trues = []
            for i in tqdm(test_dataset):
                context = parse_facts_rules(i[0], i[1])
                y_pred = model.predict(context, i[2], device=DEVICE)
                y_true = PROOFWRITER_LABEL_TO_ID[str(i[3])]
                y_preds.append(y_pred)
                y_trues.append(y_true)
            out.write(str(accuracy_score(y_trues, y_preds)))
