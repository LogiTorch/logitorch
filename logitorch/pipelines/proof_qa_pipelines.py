import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data.dataloader import DataLoader

from logitorch.data_collators.ruletaker_collator import RuleTakerCollator
from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
from logitorch.pipelines.exceptions import ModelNotCompatibleError
from logitorch.pl_models.ruletaker import PLRuleTaker

RULETAKER_COMPATIBLE_MODELS = PLRuleTaker
