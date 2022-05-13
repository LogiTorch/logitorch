# TorchTextLogic

TorchTextLogic is a PyTorch-based library for logical reasoning in natural language, it consists of:

- Textual logical reasoning datasets
- Implementations of different logical reasoning neural architectures
- A simple and clean API that can be used with PyTorch Lightning

## 📦 Installation

## 📖 Documentation

## 🖥️ Features

### 📋 Datasets

Datasets implemented in TorchTextLogic:

- [x] [AR-LSAT](https://arxiv.org/abs/2104.06598)
- [x] [ConTRoL](https://arxiv.org/abs/2011.04864)
- [x] [LogiQA](https://arxiv.org/abs/2007.08124)
- [x] [ReClor](https://arxiv.org/abs/2002.04326)
- [x] [RuleTaker](https://arxiv.org/abs/2002.05867)
- [x] [ProofWriter](https://arxiv.org/abs/2012.13048)
- [x] [SNLI](https://arxiv.org/abs/1508.05326)
- [x] [MultiNLI](https://arxiv.org/abs/1704.05426)
- [x] [RTE](https://tac.nist.gov/publications/2010/additional.papers/RTE6_overview.proceedings.pdf)
- [ ] [Negated SNLI](https://aclanthology.org/2020.emnlp-main.732/)
- [ ] [Negated MultiNLI](https://aclanthology.org/2020.emnlp-main.732/)
- [ ] [Negated RTE](https://aclanthology.org/2020.emnlp-main.732/)
  
### 🤖 Models

Models implemented in TorchTextLogic:

- [x]  [RuleTaker](https://arxiv.org/abs/2002.05867)
- [x]  [ProofWriter](https://arxiv.org/abs/2012.13048)
- [ ]  [PRover](https://arxiv.org/abs/2010.02830)
- [ ]  [FaiRR](https://arxiv.org/abs/2203.10261)
- [ ]  [LReasoner](https://arxiv.org/abs/2105.03659)
- [ ]  [DAGN](https://arxiv.org/abs/2103.14349)
- [ ]  [Focal Reasoner](https://arxiv.org/abs/2105.10334)
- [ ]  [AdaLoGN](https://arxiv.org/abs/2203.08992)
- [ ]  [Logiformer](https://arxiv.org/abs/2205.00731)
  
## 🧪 Example Usage

```python
from torchtextlogic.datasets.reclor_dataset import ReClorDataset
from torchtextlogic.pl_models.pl_ruletaker import PLRuleTaker
from torchtextlogic.data_collators.ruletaker_collator import RuleTakerCollator
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

dataset = RuleTakerDataset("depth-0", "train")
ruletaker_collate_fn = RuleTakerCollator("roberta-base")
model = PLRuleTaker("roberta-base")
train_dataloader = DataLoader(dataset, 3, collate_fn=ruletaker_collate_fn)
trainer = pl.Trainer(accelerator="gpu", gpus=1)
trainer.fit(model, train_dataloader)    
```
