Pipelines
=========

Pipelines provide pre-configured training workflows for common dataset and model combinations,
simplifying the process of training and evaluating logical reasoning models.

Overview
--------

Located in ``logitorch.pipelines``, these functions handle the complete training lifecycle including
dataset loading, model training, checkpointing, and evaluation with minimal configuration.

.. automodule:: logitorch.pipelines
   :members:
   :undoc-members:
   :show-inheritance:

Available Pipelines
-------------------

QA Pipelines
^^^^^^^^^^^^

Pre-configured pipelines for question answering tasks.

RuleTaker Pipeline
""""""""""""""""""

Train models on the RuleTaker dataset:

.. code-block:: python

   from logitorch.pipelines.qa_pipelines import ruletaker_pipeline
   from logitorch.pl_models.ruletaker import PLRuleTaker

   model = PLRuleTaker(learning_rate=1e-5, weight_decay=0.1)

   ruletaker_pipeline(
       model=model,
       dataset_name="depth-5",
       saved_model_path="models/",
       saved_model_name="best_ruletaker",
       batch_size=32,
       epochs=10,
       accelerator="gpu",
       devices=1
   )

ProofWriter Pipeline
""""""""""""""""""""

Train models on the ProofWriter dataset:

.. code-block:: python

   from logitorch.pipelines.proof_qa_pipelines import proofwriter_pipeline
   from logitorch.pl_models.proofwriter import PLProofWriter

   model = PLProofWriter(learning_rate=1e-5, weight_decay=0.1)

   proofwriter_pipeline(
       model=model,
       dataset_name="depth-5",
       saved_model_path="models/",
       saved_model_name="best_proofwriter",
       batch_size=16,
       epochs=10,
       accelerator="gpu",
       devices=1
   )

FLD Pipeline
""""""""""""

Train models on the FLD (Fine-tuned Language Decoder) dataset:

.. code-block:: python

   from logitorch.pipelines.proof_qa_pipelines import fld_pipeline
   from logitorch.pl_models.fld import PLFLDAllAtOnceProver

   model = PLFLDAllAtOnceProver(learning_rate=1e-5, weight_decay=0.1)

   fld_pipeline(
       model=model,
       dataset_name="FLD",
       saved_model_path="models/",
       saved_model_name="best_fld",
       batch_size=16,
       epochs=10,
       accelerator="gpu",
       devices=1
   )

Pipeline Parameters
-------------------

Common Parameters
^^^^^^^^^^^^^^^^^

All pipelines accept these common parameters:

- **model**: PyTorch Lightning model instance
- **saved_model_path**: Directory to save checkpoints
- **saved_model_name**: Name for the checkpoint file
- **batch_size**: Training batch size
- **epochs**: Number of training epochs
- **accelerator**: Training accelerator ("gpu", "cpu", "tpu")
- **devices**: Number of devices to use

Optional Parameters
^^^^^^^^^^^^^^^^^^^

Additional configuration options:

- **learning_rate**: Override model's default learning rate
- **weight_decay**: Override model's default weight decay
- **accumulate_grad_batches**: Gradient accumulation steps
- **gradient_clip_val**: Maximum gradient norm for clipping
- **val_check_interval**: Validation frequency
- **early_stopping_patience**: Early stopping patience

Dataset-Specific Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some pipelines accept dataset-specific parameters:

.. code-block:: python

   # RuleTaker with different depth
   ruletaker_pipeline(
       model=model,
       dataset_name="depth-3",  # or "depth-5", "depth-0", etc.
       ...
   )

   # ProofWriter with different splits
   proofwriter_pipeline(
       model=model,
       dataset_name="OWA",  # or "CWA"
       ...
   )

Usage Examples
--------------

Basic Training
^^^^^^^^^^^^^^

Train a model with default settings:

.. code-block:: python

   from logitorch.pipelines.qa_pipelines import ruletaker_pipeline
   from logitorch.pl_models.ruletaker import PLRuleTaker

   model = PLRuleTaker()

   ruletaker_pipeline(
       model=model,
       dataset_name="depth-5",
       saved_model_path="checkpoints/",
       saved_model_name="ruletaker_model",
       batch_size=32,
       epochs=10,
       accelerator="gpu",
       devices=1
   )

Advanced Configuration
^^^^^^^^^^^^^^^^^^^^^^

Customize training with advanced parameters:

.. code-block:: python

   from logitorch.pipelines.qa_pipelines import ruletaker_pipeline
   from logitorch.pl_models.ruletaker import PLRuleTaker

   model = PLRuleTaker(
       learning_rate=2e-5,
       weight_decay=0.01
   )

   ruletaker_pipeline(
       model=model,
       dataset_name="depth-5",
       saved_model_path="checkpoints/",
       saved_model_name="ruletaker_advanced",
       batch_size=16,
       epochs=20,
       accelerator="gpu",
       devices=2,  # Multi-GPU training
       accumulate_grad_batches=4,  # Effective batch size: 16 * 4 = 64
       gradient_clip_val=1.0,
       val_check_interval=0.5,  # Validate twice per epoch
       early_stopping_patience=3
   )

Multi-GPU Training
^^^^^^^^^^^^^^^^^^

Scale training across multiple GPUs:

.. code-block:: python

   ruletaker_pipeline(
       model=model,
       dataset_name="depth-5",
       saved_model_path="checkpoints/",
       saved_model_name="ruletaker_multigpu",
       batch_size=32,  # Per-device batch size
       epochs=10,
       accelerator="gpu",
       devices=4,  # Use 4 GPUs
       strategy="ddp"  # Distributed data parallel
   )

Pipeline Outputs
----------------

Training Results
^^^^^^^^^^^^^^^^

Pipelines save checkpoints to the specified directory:

.. code-block:: text

   checkpoints/
   ├── best_ruletaker_model.ckpt    # Best model checkpoint
   ├── last.ckpt                     # Last epoch checkpoint
   └── epoch=XX-step=YYYY.ckpt       # Periodic checkpoints

Best Practices
--------------

1. **Start small**: Begin with small batch sizes and short training runs to verify setup
2. **Monitor GPU memory**: Adjust batch size based on available memory
3. **Use gradient accumulation**: Achieve larger effective batch sizes without OOM errors
4. **Enable checkpointing**: Always save model checkpoints during training
5. **Validate regularly**: Set appropriate ``val_check_interval`` for your dataset size
6. **Use early stopping**: Prevent overfitting with early stopping callbacks
7. **Log metrics**: Track training progress with tools like TensorBoard or Weights & Biases
