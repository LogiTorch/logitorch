Datasets
========

LogiTorch provides a comprehensive collection of logical reasoning datasets organized by task type.

Overview
--------

The datasets module contains implementations of various benchmark datasets for logical reasoning tasks,
including question answering (QA), multiple-choice question answering (MCQA), textual entailment (TE),
proof generation, and masked language modeling (MLM).

Dataset Categories
------------------

Question Answering (QA)
^^^^^^^^^^^^^^^^^^^^^^^

Located in ``logitorch.datasets.qa``, these datasets focus on answering questions based on logical reasoning:

- **RuleTaker** - Rule-based reasoning with different depth levels
- **AbductionRules** - Abductive reasoning tasks
- **ParaRules Plus** - Enhanced paragraph-based rule reasoning

.. automodule:: logitorch.datasets.qa
   :members:
   :undoc-members:
   :show-inheritance:

Multiple-Choice Question Answering (MCQA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Located in ``logitorch.datasets.mcqa``, these datasets present questions with multiple answer choices:

- **LogiQA** - Logical reasoning questions
- **LogiQA 2.0** - Updated version with more challenging questions
- **ReClor** - Reading comprehension with logical reasoning
- **AR-LSAT** - Analytical reasoning from LSAT exams

.. automodule:: logitorch.datasets.mcqa
   :members:
   :undoc-members:
   :show-inheritance:

Textual Entailment (TE)
^^^^^^^^^^^^^^^^^^^^^^^^

Located in ``logitorch.datasets.te``, these datasets focus on determining logical relationships between text pairs:

- **SNLI** - Stanford Natural Language Inference
- **MultiNLI** - Multi-Genre Natural Language Inference
- **RTE** - Recognizing Textual Entailment
- **Negated SNLI/MultiNLI/RTE** - Negated versions for robustness testing
- **ConTRoL** - Controlled reasoning over text
- **LogiQA2NLI** - LogiQA converted to NLI format
- **FOLIO** - First-Order Logic Inference

.. automodule:: logitorch.datasets.te
   :members:
   :undoc-members:
   :show-inheritance:

Proof Generation (proof_qa)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Located in ``logitorch.datasets.proof_qa``, these datasets require generating logical proofs:

- **ProofWriter** - Generating natural language proofs
- **FLD** - Forward Logic Deduction

.. automodule:: logitorch.datasets.proof_qa
   :members:
   :undoc-members:
   :show-inheritance:

Masked Language Modeling (MLM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Located in ``logitorch.datasets.mlm``, these datasets are designed for pre-training with masked language modeling:

.. automodule:: logitorch.datasets.mlm
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Here's how to use a dataset:

.. code-block:: python

   from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset

   # Load training dataset
   train_dataset = RuleTakerDataset("depth-5", "train")

   # Access a sample
   sample = train_dataset[0]
   print(sample["context"])
   print(sample["question"])
   print(sample["label"])

For textual entailment:

.. code-block:: python

   from logitorch.datasets.te.snli_dataset import SNLIDataset

   # Load dataset
   dataset = SNLIDataset("train")

   # Access a sample
   sample = dataset[0]
   print(sample["premise"])
   print(sample["hypothesis"])
   print(sample["label"])

Dataset Classes
---------------

Base Dataset
^^^^^^^^^^^^

All datasets inherit from PyTorch's ``Dataset`` class and follow a consistent interface.

.. code-block:: python

   from torch.utils.data import Dataset

   class LogicDataset(Dataset):
       def __init__(self, split: str):
           """
           Args:
               split: Dataset split ('train', 'val', 'test')
           """
           pass

       def __len__(self):
           """Returns the number of samples in the dataset"""
           pass

       def __getitem__(self, idx):
           """Returns a single sample as a dictionary"""
           pass
