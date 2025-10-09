Data Collators
==============

Data collators are responsible for preparing batches of data for training and evaluation. They handle
tokenization, padding, and formatting of inputs for specific models.

Overview
--------

Located in ``logitorch.data_collators``, these classes process raw dataset samples into batched tensors
suitable for model training. Each collator is typically paired with a specific dataset and model combination.

.. automodule:: logitorch.data_collators
   :members:
   :undoc-members:
   :show-inheritance:

Available Collators
-------------------

RuleTaker Collator
^^^^^^^^^^^^^^^^^^

Processes RuleTaker dataset samples for the RuleTaker model.

.. code-block:: python

   from logitorch.data_collators.ruletaker_collator import RuleTakerCollator
   from torch.utils.data import DataLoader

   collator = RuleTakerCollator(
       model_name="bert-base-uncased",
       max_length=512
   )

   dataloader = DataLoader(
       dataset,
       batch_size=32,
       collate_fn=collator
   )

ProofWriter Collator
^^^^^^^^^^^^^^^^^^^^

Processes ProofWriter dataset samples for proof generation tasks.

.. code-block:: python

   from logitorch.data_collators.proofwriter_collator import ProofWriterCollator

   collator = ProofWriterCollator(
       model_name="t5-base",
       max_length=512
   )

BERTNOT Collator
^^^^^^^^^^^^^^^^

Processes samples with special handling for negation tokens.

.. code-block:: python

   from logitorch.data_collators.bertnot_collator import BERTNOTCollator

   collator = BERTNOTCollator(
       model_name="bert-base-uncased",
       max_length=512
   )

PRover Collator
^^^^^^^^^^^^^^^

Processes samples for the PRover model with rule-based attention.

.. code-block:: python

   from logitorch.data_collators.prover_collator import PROVERCollator

   collator = PROVERCollator(
       model_name="bert-base-uncased",
       max_length=512
   )

FLD Collator
^^^^^^^^^^^^

Processes samples for forward logic deduction tasks.

.. code-block:: python

   from logitorch.data_collators.fld_collator import FLDCollator

   collator = FLDCollator(
       model_name="bert-base-uncased",
       max_length=512
   )

FaiRR Collator
^^^^^^^^^^^^^^

Processes samples for the FaiRR (Faithful and Robust Reasoning) model.

.. code-block:: python

   from logitorch.data_collators.fairr_collator import FaiRRCollator

   collator = FaiRRCollator(
       model_name="bert-base-uncased",
       max_length=512
   )

DAGN Collator
^^^^^^^^^^^^^

Processes samples for the DAGN (Differential Attention Graph Network) model.

.. code-block:: python

   from logitorch.data_collators.dagn_collator import DAGNCollator

   collator = DAGNCollator(
       model_name="bert-base-uncased",
       max_length=512
   )

Usage Guide
-----------

Basic Usage
^^^^^^^^^^^

Data collators are used with PyTorch's ``DataLoader`` to batch and prepare data:

.. code-block:: python

   from torch.utils.data import DataLoader
   from logitorch.datasets.qa.ruletaker_dataset import RuleTakerDataset
   from logitorch.data_collators.ruletaker_collator import RuleTakerCollator

   # Create dataset
   dataset = RuleTakerDataset("depth-5", "train")

   # Create collator
   collator = RuleTakerCollator()

   # Create dataloader
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       collate_fn=collator,
       shuffle=True
   )

   # Iterate over batches
   for batch in dataloader:
       # batch contains tokenized and padded inputs
       input_ids = batch["input_ids"]
       attention_mask = batch["attention_mask"]
       labels = batch["labels"]

Custom Tokenization
^^^^^^^^^^^^^^^^^^^

You can customize tokenization parameters:

.. code-block:: python

   collator = RuleTakerCollator(
       model_name="roberta-large",
       max_length=1024,
       padding="max_length",
       truncation=True
   )

Collator Interface
------------------

Base Collator
^^^^^^^^^^^^^

All collators follow a consistent interface:

.. code-block:: python

   from transformers import AutoTokenizer
   from typing import List, Dict, Any

   class BaseCollator:
       def __init__(
           self,
           model_name: str = "bert-base-uncased",
           max_length: int = 512,
           padding: str = "max_length",
           truncation: bool = True
       ):
           """
           Args:
               model_name: HuggingFace model name for tokenizer
               max_length: Maximum sequence length
               padding: Padding strategy ('max_length', 'longest', 'do_not_pad')
               truncation: Whether to truncate sequences
           """
           self.tokenizer = AutoTokenizer.from_pretrained(model_name)
           self.max_length = max_length
           self.padding = padding
           self.truncation = truncation

       def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
           """
           Collate a batch of samples.

           Args:
               batch: List of dataset samples

           Returns:
               Dictionary containing batched tensors
           """
           pass

Output Format
^^^^^^^^^^^^^

Collators typically return a dictionary with the following keys:

- **input_ids**: Token IDs for the input sequence
- **attention_mask**: Mask indicating which tokens are padding
- **token_type_ids**: Segment IDs (for models like BERT)
- **labels**: Target labels or output sequences
- **Additional keys**: Model-specific inputs (e.g., rule attention masks)

Common Parameters
-----------------

Model Name
^^^^^^^^^^

Specifies the pretrained model/tokenizer to use:

.. code-block:: python

   # Using BERT
   collator = RuleTakerCollator(model_name="bert-base-uncased")

   # Using RoBERTa
   collator = RuleTakerCollator(model_name="roberta-large")

   # Using T5
   collator = ProofWriterCollator(model_name="t5-base")

Max Length
^^^^^^^^^^

Controls the maximum sequence length:

.. code-block:: python

   # Shorter sequences for faster training
   collator = RuleTakerCollator(max_length=256)

   # Longer sequences for complex reasoning
   collator = RuleTakerCollator(max_length=1024)

Padding Strategy
^^^^^^^^^^^^^^^^

Controls how sequences are padded:

.. code-block:: python

   # Pad to max_length (uniform batch size)
   collator = RuleTakerCollator(padding="max_length")

   # Pad to longest in batch (more efficient)
   collator = RuleTakerCollator(padding="longest")

Best Practices
--------------

1. **Match collator to model**: Use the collator designed for your specific model architecture
2. **Optimize max_length**: Balance between capturing full context and memory/speed
3. **Use dynamic padding**: Set ``padding="longest"`` for better efficiency during training
4. **Batch size tuning**: Adjust batch size based on max_length and available GPU memory
