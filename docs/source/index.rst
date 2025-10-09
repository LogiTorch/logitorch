.. LogiTorch documentation master file, created by
   sphinx-quickstart on Sun Nov 21 00:08:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LogiTorch's documentation!
==========================================

LogiTorch is a PyTorch-based library for logical reasoning on natural language. It provides:

- **Textual logical reasoning datasets** - Access to numerous benchmark datasets for logical reasoning tasks
- **Neural architecture implementations** - State-of-the-art models for logical reasoning
- **Clean PyTorch Lightning API** - Simple and extensible interface for training and evaluation

Installation
------------

Install LogiTorch using pip:

.. code-block:: console

   pip install logitorch

Or install from source:

.. code-block:: console

   pip install git+https://github.com/LogiTorch/logitorch.git

Quick Start
-----------

Here's a simple example to get started with LogiTorch:

.. code-block:: python

   import pytorch_lightning as pl
   from torch.utils.data.dataloader import DataLoader

   from logitorch.data_collators.ruletaker_collator import RuleTakerCollator
   from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
   from logitorch.pl_models.ruletaker import PLRuleTaker

   # Load datasets
   train_dataset = RuleTakerDataset("depth-5", "train")
   val_dataset = RuleTakerDataset("depth-5", "val")

   # Create data loaders
   collate_fn = RuleTakerCollator()
   train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
   val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

   # Initialize model
   model = PLRuleTaker(learning_rate=1e-5, weight_decay=0.1)

   # Train
   trainer = pl.Trainer(accelerator="gpu", devices=1)
   trainer.fit(model, train_dataloader, val_dataloader)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   datasets
   models
   data_collators
   pipelines
   losses
   utilities


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
