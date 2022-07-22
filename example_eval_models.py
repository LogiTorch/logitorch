from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torchtextlogic.data_collators.proofwriter_collator import (
    ProofWriterProofGenerationAllCollator,
)
from torchtextlogic.data_collators.prover_collator import PRoverProofWriterCollator
from torchtextlogic.data_collators.ruletaker_collator import (
    RuleTakerProofWriterCollator,
)
from torchtextlogic.datasets.proof_qa.proofwriter_dataset import (
    PROOFWRITER_LABEL_TO_ID,
    ProofWriterDataset,
)
from torchtextlogic.pl_models.proofwriter import PLProofWriter
from torchtextlogic.pl_models.prover import PLPRover
from torchtextlogic.pl_models.ruletaker import PLRuleTaker

MODEL = "proofwriter"
DEVICE = "cpu"


def parse_facts_rules(facts, rules):
    sentences = []
    for s in facts.values():
        sentences.append(s)
    for s in rules.values():
        sentences.append(s)
    context = "".join(sentences)
    return context


if MODEL == "proofwriter":
    model = PLProofWriter.load_from_checkpoint(
        "models/best_proofwriter-epoch=04-val_loss=0.20.ckpt",
        pretrained_model="t5-large",
    )
    model.to(DEVICE)
    model.eval()
    for d in range(0, 6):
        if d == 4:
            d = "3ext"
        test_dataset = ProofWriterDataset(f"depth-{d}", "val", "proof_generation_all")
        with open(f"proofwriter_{d}.txt", "w") as out:
            y_preds = []
            y_trues = []
            for i in test_dataset:
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
        "models/best_prover-epoch=00-val_loss=0.52.ckpt",
        pretrained_model="roberta-large",
    )
    model.to(DEVICE)
    model.eval()
    for d in range(0, 6):
        if d == 4:
            d = "3ext"
        test_dataset = ProofWriterDataset(f"depth-{d}", "test", "proof_generation_all")
        with open(f"prover_{d}.txt", "w") as out:
            y_preds = []
            y_trues = []
            for i in test_dataset:
                y_pred = model.predict(i[0], i[1], i[2], device=DEVICE)
                y_true = PROOFWRITER_LABEL_TO_ID[str(i[3])]
                y_preds.append(y_pred)
                y_trues.append(y_true)
            out.write(str(accuracy_score(y_trues, y_preds)))

elif MODEL == "ruletaker":
    test_dataset = ProofWriterDataset("depth-0", "test", "proof_generation_all")
    model = PLRuleTaker.load_from_checkpoint(
        "models/best_ruletaker-epoch=03-val_loss=0.69.ckpt",
        pretrained_model="roberta-large",
    )
    model.to(DEVICE)
    model.eval()
    for d in range(0, 6):
        if d == 4:
            d = "3ext"
        test_dataset = ProofWriterDataset(f"depth-{d}", "test", "proof_generation_all")
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
