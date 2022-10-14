# LogiTorch

![image](docs/assets/logo.jpg)

LogiTorch is a PyTorch-based library for logical reasoning on natural language, it consists of:

- Textual logical reasoning datasets
- Implementations of different logical reasoning neural architectures
- A simple and clean API that can be used with PyTorch Lightning

## 📦 Installation

```console
foo@bar:~$ pip install logitorch==0.0.1a1
```

## 📖 Documentation

You can find the documentation for LogiTorch on [ReadTheDocs](https://logitorch.readthedocs.io).

## 🖥️ Features

### 📋 Datasets

Datasets implemented in LogiTorch:

- [x] [AR-LSAT](https://arxiv.org/abs/2104.06598) <sub>(MIT LICENSE)</sub>
- [x] [ConTRoL](https://arxiv.org/abs/2011.04864) <sub>(GitHub LICENSE)</sub>
- [x] [LogiQA](https://arxiv.org/abs/2007.08124) <sub>(GitHub LICENSE)</sub>
- [x] [ReClor](https://arxiv.org/abs/2002.04326) <sub>(Non-Commercial Research Use)</sub>
- [x] [RuleTaker](https://arxiv.org/abs/2002.05867) <sub>(APACHE-2.0 LICENSE)</sub>
- [x] [ProofWriter](https://arxiv.org/abs/2012.13048) <sub>(APACHE-2.0 LICENSE)</sub>
- [x] [SNLI](https://arxiv.org/abs/1508.05326) <sub>(CC-BY-SA-4.0 LICENSE)</sub>
- [x] [MultiNLI](https://arxiv.org/abs/1704.05426) <sub>(CC-BY-SA-4.0 LICENSE)</sub>
- [x] [RTE](https://tac.nist.gov/publications/2010/additional.papers/RTE6_overview.proceedings.pdf) <sub>([TAC User Agreements](https://tac.nist.gov//data/forms/index.html))</sub>
- [x] [Negated SNLI](https://aclanthology.org/2020.emnlp-main.732/) <sub>(MIT LICENSE)</sub>
- [x] [Negated MultiNLI](https://aclanthology.org/2020.emnlp-main.732/) <sub>(MIT LICENSE)</sub>
- [x] [Negated RTE](https://aclanthology.org/2020.emnlp-main.732/) <sub>(MIT LICENSE)</sub>
- [x] [PARARULES Plus](https://github.com/Strong-AI-Lab/PARARULE-Plus) <sub>(MIT LICENSE)</sub>
- [x] [AbductionRules](https://arxiv.org/abs/2203.12186) <sub>(MIT LICENSE)</sub>
- [x] [FOLIO](https://arxiv.org/abs/2209.00840) <sub>(CC-BY-SA-4.0 LICENSE)</sub>
- [ ] [SimpleLogic](https://arxiv.org/abs/2205.11502)
- [ ] [RobustLR](https://arxiv.org/abs/2205.12598)
- [ ] [LogicNLI](https://aclanthology.org/2021.emnlp-main.303/)

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
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from logitorch.data_collators.ruletaker_collator import RuleTakerCollator
from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
from logitorch.pl_models.ruletaker import PLRuleTaker

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
    filename="best_ruletaker",
)

trainer = pl.Trainer(accelerator="gpu", gpus=1)
trainer.fit(model, train_dataloader, val_dataloader)
```

### Testing Example

```python
from logitorch.pl_models.ruletaker import PLRuleTaker
from logitorch.datasets.qa.ruletaker_dataset import RULETAKER_ID_TO_LABEL

model = PLRuleTaker.load_from_checkpoint("best_ruletaker.ckpt")

context = "Bob is smart. If someone is smart then he is kind."
question = "Bob is kind."

pred = model.predict(context, question)
print(RULETAKER_ID_TO_LABEL[pred])
```

## Citing

Users of LogiTorch should distinguish the datasets and models of our library from the originals. They should always credit and cite both our library and the original data source, as in ``We used LogiTorch's (citation) re-implementation of BERTNOT \cite{hosseini2021understanding}''.

If you want to cite logitorch, please refer to the publication in the [Empirical Methods in Natural Language Processing](https://2022.emnlp.org/):

```code
@inproceedings{helwe2022logitorch,
  title={LogiTorch: A Pytorch-based library for logical reasoning on natural language},
  author={Helwe, Chadi and Clavel, Chlo\'e and Suchanek, Fabian},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year={2022}
}
```

## Acknowledgments

This work was partially funded by ANR-20-CHIA-0012-01 (“NoRDF”).
