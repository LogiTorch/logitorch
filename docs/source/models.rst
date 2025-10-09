Models
======

LogiTorch provides implementations of state-of-the-art neural architectures for logical reasoning tasks.

Overview
--------

The models module contains both standard PyTorch model implementations and PyTorch Lightning wrappers
for easy training and evaluation. All models are designed to work seamlessly with LogiTorch datasets
and data collators.

Model Categories
----------------

PyTorch Models
^^^^^^^^^^^^^^

Located in ``logitorch.models``, these are standard PyTorch model implementations:

.. automodule:: logitorch.models
   :members:
   :undoc-members:
   :show-inheritance:

PyTorch Lightning Models
^^^^^^^^^^^^^^^^^^^^^^^^^

Located in ``logitorch.pl_models``, these models extend PyTorch Lightning's ``LightningModule``
for simplified training workflows:

.. automodule:: logitorch.pl_models
   :members:
   :undoc-members:
   :show-inheritance:

Available Models
----------------

RuleTaker
^^^^^^^^^

A model designed for rule-based reasoning tasks that can handle different depths of logical inference.

**Reference:** `Transformers as Soft Reasoners over Language <https://arxiv.org/abs/2002.05867>`_

.. code-block:: python

   from logitorch.pl_models.ruletaker import PLRuleTaker

   model = PLRuleTaker(
       learning_rate=1e-5,
       weight_decay=0.1
   )

ProofWriter
^^^^^^^^^^^

A model that generates natural language proofs for logical reasoning questions.

**Reference:** `ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language <https://arxiv.org/abs/2012.13048>`_

.. code-block:: python

   from logitorch.pl_models.proofwriter import PLProofWriter

   model = PLProofWriter(
       learning_rate=1e-5,
       weight_decay=0.1
   )

BERTNOT
^^^^^^^

A model that incorporates explicit negation handling for improved logical reasoning.

**Reference:** `Understanding by Understanding Not: Modeling Negation in Language Models <https://arxiv.org/abs/2105.03519>`_

.. code-block:: python

   from logitorch.pl_models.bertnot import PLBERTNOT

   model = PLBERTNOT(
       learning_rate=1e-5,
       weight_decay=0.1
   )

PRover
^^^^^^

A model that performs proof generation using attention mechanisms over logical rules.

**Reference:** `PRover: Proof Generation for Interpretable Reasoning over Rules <https://arxiv.org/abs/2010.02830>`_

.. code-block:: python

   from logitorch.pl_models.prover import PLPRover

   model = PLPRover(
       learning_rate=1e-5,
       weight_decay=0.1
   )

FLD (Fine-tuned Language Decoder)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A model designed for logical reasoning through fine-tuned language decoding.

**Reference:** `FLD: Towards Improving Deep Neural Network Reasoning <https://proceedings.mlr.press/v202/morishita23a.html>`_

.. code-block:: python

   from logitorch.pl_models.fld import PLFLDAllAtOnceProver

   model = PLFLDAllAtOnceProver(
       learning_rate=1e-5,
       weight_decay=0.1
   )

Model Usage
-----------

Training a Model
^^^^^^^^^^^^^^^^

All PyTorch Lightning models follow a consistent training interface:

.. code-block:: python

   import pytorch_lightning as pl
   from torch.utils.data import DataLoader
   from logitorch.pl_models.ruletaker import PLRuleTaker
   from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
   from logitorch.data_collators.ruletaker_collator import RuleTakerCollator

   # Create datasets
   train_dataset = RuleTakerDataset("depth-5", "train")
   val_dataset = RuleTakerDataset("depth-5", "val")

   # Create data loaders
   collate_fn = RuleTakerCollator()
   train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
   val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

   # Initialize model
   model = PLRuleTaker(learning_rate=1e-5, weight_decay=0.1)

   # Train
   trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10)
   trainer.fit(model, train_loader, val_loader)

Making Predictions
^^^^^^^^^^^^^^^^^^

After training, you can use models for inference:

.. code-block:: python

   from logitorch.pl_models.ruletaker import PLRuleTaker
   from logitorch.datasets.qa.ruletaker_dataset import RULETAKER_ID_TO_LABEL

   # Load trained model
   model = PLRuleTaker.load_from_checkpoint("path/to/checkpoint.ckpt")

   # Make prediction
   context = "Bob is smart. If someone is smart then he is kind."
   question = "Bob is kind."

   prediction = model.predict(context, question)
   label = RULETAKER_ID_TO_LABEL[prediction]
   print(f"Prediction: {label}")

Model Configuration
-------------------

Common Parameters
^^^^^^^^^^^^^^^^^

Most PyTorch Lightning models in LogiTorch support these common parameters:

- **learning_rate** (float): Learning rate for the optimizer (default: 1e-5)
- **weight_decay** (float): Weight decay for regularization (default: 0.1)
- **model_name** (str): Pretrained model name from HuggingFace (default: "bert-base-uncased")
- **num_labels** (int): Number of output labels for classification tasks

Model Architecture
------------------

Base Classes
^^^^^^^^^^^^

All models follow a consistent architecture pattern:

.. code-block:: python

   import pytorch_lightning as pl
   from torch import nn

   class LogicReasoningModel(pl.LightningModule):
       def __init__(self, learning_rate: float = 1e-5, weight_decay: float = 0.1):
           super().__init__()
           self.save_hyperparameters()

       def forward(self, **inputs):
           """Forward pass through the model"""
           pass

       def training_step(self, batch, batch_idx):
           """Single training step"""
           pass

       def validation_step(self, batch, batch_idx):
           """Single validation step"""
           pass

       def configure_optimizers(self):
           """Configure optimizer and learning rate scheduler"""
           pass

       def predict(self, context: str, question: str):
           """Make a prediction for a single example"""
           pass
