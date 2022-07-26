# LogiTorch

LogiTorch is a PyTorch-based library for logical reasoning in natural language, it consists of:

- Textual logical reasoning datasets
- Implementations of different logical reasoning neural architectures
- A simple and clean API that can be used with PyTorch Lightning

## 📦 Installation

## 📖 Documentation

## 🖥️ Features

### 📋 Datasets

Datasets implemented in LogiTorch:

- [x] [AR-LSAT](https://arxiv.org/abs/2104.06598)
- [x] [ConTRoL](https://arxiv.org/abs/2011.04864)
- [x] [LogiQA](https://arxiv.org/abs/2007.08124)
- [x] [ReClor](https://arxiv.org/abs/2002.04326)
- [x] [RuleTaker](https://arxiv.org/abs/2002.05867)
- [x] [ProofWriter](https://arxiv.org/abs/2012.13048)
- [x] [SNLI](https://arxiv.org/abs/1508.05326)
- [x] [MultiNLI](https://arxiv.org/abs/1704.05426)
- [x] [RTE](https://tac.nist.gov/publications/2010/additional.papers/RTE6_overview.proceedings.pdf)
- [x] [Negated SNLI](https://aclanthology.org/2020.emnlp-main.732/)
- [x] [Negated MultiNLI](https://aclanthology.org/2020.emnlp-main.732/)
- [x] [Negated RTE](https://aclanthology.org/2020.emnlp-main.732/)
- [x] [PARARULES Plus](https://github.com/Strong-AI-Lab/PARARULE-Plus)
- [x] [AbductionRules](https://arxiv.org/abs/2203.12186)

### 🤖 Models

Models implemented in LogiTorch:

- [x]  [RuleTaker](https://arxiv.org/abs/2002.05867)
- [x]  [ProofWriter](https://arxiv.org/abs/2012.13048)
- [x]  [BERTNOT](https://arxiv.org/abs/2105.03519)
- [x]  [PRover](https://arxiv.org/abs/2010.02830)
- [ ]  [FaiRR](https://arxiv.org/abs/2203.10261)
- [ ]  [LReasoner](https://arxiv.org/abs/2105.03659)
- [ ]  [DAGN](https://arxiv.org/abs/2103.14349)
- [ ]  [Focal Reasoner](https://arxiv.org/abs/2105.10334)
- [ ]  [AdaLoGN](https://arxiv.org/abs/2203.08992)
- [ ]  [Logiformer](https://arxiv.org/abs/2205.00731)
- [ ]  [LogiGAN](https://arxiv.org/abs/2205.08794)

## 🧪 Example Usage

### Training Example
```python
import pytorch_lightning as pl
from logitorch.data_collators.ruletaker_collator import RuleTakerCollator
from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
from logitorch.pl_models.ruletaker import PLRuleTaker
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

train_dataset = RuleTakerDataset("depth-5", "train")
val_dataset = RuleTakerDataset("depth-5", "val")

ruletaker_collate_fn = RuleTakerCollator()
train_dataloader = DataLoader(
    train_dataset, batch_size=32, collate_fn=ruletaker_collate_fn
)

val_dataloader = DataLoader(
    train_dataset, batch_size=32, collate_fn=ruletaker_collate_fn
)

model = PLRuleTaker(learning_rate=1e-5, weight_decay=0.1)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath="models/",
    filename="best_ruletaker.ckpt",
)

trainer = pl.Trainer(accelerator="gpu", gpus=1)
trainer.fit(model, train_dataloader, val_dataloader)
```

### Testing Example
```python
from logitorch.pl_models.ruletaker import PLRuleTaker
from logitorch.datasets.qa.ruletaker_dataset import RULETAKER_ID_TO_LABEL
import pytorch_lightning as pl

model = PLRuleTaker.load_from_checkpoint("best_ruletaker.ckpt")

context = "Bob is smart. If someone is smart then he is kind"
question = "Bob is kind"

pred = model.predict(context, question)
print(RULETAKER_ID_TO_LABEL[pred])
```
