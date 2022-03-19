# TorchTextLogic

TorchTextLogic is a PyTorch-based library for logical reasoning in natural language, it consists of:
- Textual logical reasoning datasets
- Implementations of different logical reasoning neural architectures
- A simple and clean API that can be used with PyTorch Lightning

## Installation

## Documentation

## Example Usage

```python
from torchtextlogic.datasets.reclor_dataset import ReClorDataset
from torchtextlogic.pl_models.pl_ruletaker import PLRuleTaker
from torchtextlogic.data_collator.ruletaker_collator import RuleTakerCollator
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

dataset = RuleTakerDataset("depth-0", "train")
ruletaker_collate_fn = RuleTakerCollator("roberta-base")
model = PLRuleTaker("roberta-base")
train_dataloader = DataLoader(dataset, 3, collate_fn=ruletaker_collate_fn)
trainer = pl.Trainer(accelerator="gpu", gpus=1)
trainer.fit(model, train_dataloader)    
```

