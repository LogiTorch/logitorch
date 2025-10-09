Losses
======

Custom loss functions designed for logical reasoning tasks.

Overview
--------

Located in ``logitorch.losses``, these loss functions provide specialized training objectives
for logical reasoning models beyond standard cross-entropy loss.

.. automodule:: logitorch.losses
   :members:
   :undoc-members:
   :show-inheritance:

Available Loss Functions
------------------------

Unlikelihood Loss
^^^^^^^^^^^^^^^^^

The unlikelihood loss is designed to explicitly penalize incorrect predictions by reducing
their probability during training. This is particularly useful for logical reasoning tasks
where certain predictions should be strongly discouraged.

**Reference:** `Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases <https://arxiv.org/abs/1909.03683>`_

.. code-block:: python

   from logitorch.losses.unlikelihood_loss import UnlikelihoodLoss

   # Initialize loss function
   loss_fn = UnlikelihoodLoss()

   # Compute loss
   loss = loss_fn(logits, targets)

Usage Examples
--------------

Basic Usage
^^^^^^^^^^^

Using unlikelihood loss in a training loop:

.. code-block:: python

   import torch
   from logitorch.losses.unlikelihood_loss import UnlikelihoodLoss

   # Initialize model and loss
   model = YourModel()
   loss_fn = UnlikelihoodLoss()
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

   # Training loop
   for batch in dataloader:
       optimizer.zero_grad()

       # Forward pass
       logits = model(batch["input_ids"], batch["attention_mask"])

       # Compute loss
       loss = loss_fn(logits, batch["labels"])

       # Backward pass
       loss.backward()
       optimizer.step()

With PyTorch Lightning
^^^^^^^^^^^^^^^^^^^^^^

Integrating custom losses in PyTorch Lightning models:

.. code-block:: python

   import pytorch_lightning as pl
   from logitorch.losses.unlikelihood_loss import UnlikelihoodLoss

   class MyModel(pl.LightningModule):
       def __init__(self):
           super().__init__()
           self.model = YourBaseModel()
           self.loss_fn = UnlikelihoodLoss()

       def training_step(self, batch, batch_idx):
           logits = self.model(batch["input_ids"], batch["attention_mask"])
           loss = self.loss_fn(logits, batch["labels"])

           self.log("train_loss", loss)
           return loss

       def configure_optimizers(self):
           return torch.optim.AdamW(self.parameters(), lr=1e-5)

Combined Loss Functions
^^^^^^^^^^^^^^^^^^^^^^^

You can combine multiple loss functions for multi-task learning:

.. code-block:: python

   import torch.nn as nn
   from logitorch.losses.unlikelihood_loss import UnlikelihoodLoss

   class CombinedLoss(nn.Module):
       def __init__(self, alpha=0.5):
           super().__init__()
           self.alpha = alpha
           self.ce_loss = nn.CrossEntropyLoss()
           self.ul_loss = UnlikelihoodLoss()

       def forward(self, logits, targets):
           ce = self.ce_loss(logits, targets)
           ul = self.ul_loss(logits, targets)
           return self.alpha * ce + (1 - self.alpha) * ul

Loss Function Interface
-----------------------

Standard Interface
^^^^^^^^^^^^^^^^^^

All loss functions follow PyTorch's standard loss interface:

.. code-block:: python

   import torch.nn as nn

   class CustomLoss(nn.Module):
       def __init__(self, reduction='mean'):
           """
           Args:
               reduction: Specifies reduction to apply to output
                         ('none', 'mean', 'sum')
           """
           super().__init__()
           self.reduction = reduction

       def forward(self, input, target):
           """
           Args:
               input: Predicted logits (batch_size, num_classes)
               target: Ground truth labels (batch_size,)

           Returns:
               Loss value (scalar or tensor depending on reduction)
           """
           pass

Parameters
----------

Common Parameters
^^^^^^^^^^^^^^^^^

Most loss functions support these parameters:

- **reduction** (str): Specifies how to reduce the loss

  - ``'none'``: No reduction, return loss per sample
  - ``'mean'``: Return mean of all losses (default)
  - ``'sum'``: Return sum of all losses

- **weight** (Tensor, optional): Manual rescaling weight for each class
- **ignore_index** (int, optional): Target value to ignore in loss computation

Example with parameters:

.. code-block:: python

   from logitorch.losses.unlikelihood_loss import UnlikelihoodLoss

   # With class weights
   class_weights = torch.tensor([1.0, 2.0, 1.5])
   loss_fn = UnlikelihoodLoss(weight=class_weights)

   # With custom reduction
   loss_fn = UnlikelihoodLoss(reduction='sum')

   # Ignoring padding tokens
   loss_fn = UnlikelihoodLoss(ignore_index=-100)

Best Practices
--------------

1. **Choose appropriate loss**: Select loss functions that match your task requirements
2. **Balance multiple losses**: When combining losses, tune the weighting coefficients
3. **Monitor loss values**: Track both training and validation losses
4. **Gradient clipping**: Use gradient clipping with custom losses to prevent instability
5. **Numerical stability**: Ensure loss computations are numerically stable

Troubleshooting
---------------

NaN or Inf Loss Values
^^^^^^^^^^^^^^^^^^^^^^

If you encounter NaN or infinite loss values:

1. Check for extreme values in logits
2. Ensure proper gradient clipping
3. Reduce learning rate
4. Add numerical stability terms (e.g., epsilon values)

.. code-block:: python

   # Add gradient clipping in PyTorch Lightning
   trainer = pl.Trainer(gradient_clip_val=1.0)

Loss Not Decreasing
^^^^^^^^^^^^^^^^^^^

If loss is not decreasing during training:

1. Verify loss function is appropriate for the task
2. Check learning rate (may be too low or too high)
3. Ensure model architecture matches task complexity
4. Verify data preprocessing and labels are correct
