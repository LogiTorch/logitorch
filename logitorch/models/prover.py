import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.init import xavier_normal_
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from logitorch.data_collators.prover_collator import PRoverProofWriterCollator


class _NodeClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        xavier_normal_(self.dense.weight)
        xavier_normal_(self.out_proj.weight)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class _EdgeClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        xavier_normal_(self.dense.weight)
        xavier_normal_(self.out_proj.weight)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PRover(nn.Module):
    def __init__(self, pretrained_roberta_model: str, num_labels: int = 2) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.num_labels_edge = num_labels
        self.proofwriter_collator = PRoverProofWriterCollator(pretrained_roberta_model)
        self.encoder = RobertaModel.from_pretrained(pretrained_roberta_model)
        self.config = self.encoder.config
        self.naf_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier = RobertaClassificationHead(self.config)
        self.classifier_node = _NodeClassificationHead(self.config)
        self.classifier_edge = _EdgeClassificationHead(self.config)

        # xavier_normal(self.naf_layer)
        # xavier_normal(self.classifier_node)
        # xavier_normal(self.classifier_edge)

    def forward(
        self,
        x,
        proof_offsets=None,
        node_labels=None,
        edge_labels=None,
        qa_labels=None,
        max_node_length=None,
        max_edge_length=None,
        device: str = "cpu",
    ):
        outputs = self.encoder(**x)
        sequence_outputs = outputs[0]
        cls_outputs = sequence_outputs[:, 0, :]
        naf_outputs = self.naf_layer(cls_outputs)
        logits = self.classifier(sequence_outputs)

        if max_node_length is None:
            max_node_length = node_labels.shape[1]

        if max_edge_length is None:
            max_edge_length = edge_labels.shape[1]

        if node_labels is None:
            batch_size = 1
        else:
            batch_size = node_labels.shape[0]
        embedding_dim = sequence_outputs.shape[2]

        # print(max_node_length)
        # print(max_edge_length)
        # print(batch_size)
        # print(embedding_dim)
        batch_node_embedding = torch.zeros(
            (batch_size, max_node_length, embedding_dim)
        ).to(device)
        batch_edge_embedding = torch.zeros(
            (batch_size, max_edge_length, 3 * embedding_dim)
        ).to(device)

        for batch_index in range(batch_size):
            prev_index = 1
            sample_node_embedding = None
            count = 0
            for offset in proof_offsets[batch_index][1:]:
                if offset == 0:
                    break
                else:
                    rf_embedding = torch.mean(
                        sequence_outputs[batch_index, prev_index : (offset + 1), :],
                        dim=0,
                    ).unsqueeze(0)
                    prev_index = offset + 1
                    count += 1
                    if sample_node_embedding is None:
                        sample_node_embedding = rf_embedding
                    else:
                        sample_node_embedding = torch.cat(
                            (sample_node_embedding, rf_embedding), dim=0
                        )

            # Add the NAF output at the end
            sample_node_embedding = torch.cat(
                (sample_node_embedding, naf_outputs[batch_index].unsqueeze(0)), dim=0
            )
            repeat1 = sample_node_embedding.unsqueeze(0).repeat(
                len(sample_node_embedding), 1, 1
            )
            repeat2 = sample_node_embedding.unsqueeze(1).repeat(
                1, len(sample_node_embedding), 1
            )
            sample_edge_embedding = torch.cat(
                (repeat1, repeat2, (repeat1 - repeat2)), dim=2
            )

            sample_edge_embedding = sample_edge_embedding.view(
                -1, sample_edge_embedding.shape[-1]
            )

            if sample_node_embedding.shape[0] < max_node_length:
                # Append 0s at the end (these will be ignored for loss)
                sample_node_embedding = torch.cat(
                    (
                        sample_node_embedding,
                        torch.zeros((max_node_length - count - 1, embedding_dim)).to(
                            device
                        ),
                    ),
                    dim=0,
                )
            # print(sample_node_embedding.shape)

            # print(max_edge_length)
            # print(sample_edge_embedding.shape[0])

            sample_edge_embedding = torch.cat(
                (
                    sample_edge_embedding,
                    torch.zeros(
                        (
                            max_edge_length - len(sample_edge_embedding),
                            3 * embedding_dim,
                        )
                    ).to(device),
                ),
                dim=0,
            )

            # print(max_edge_length)
            # print(len(sample_edge_embedding))
            # print(sample_edge_embedding.shape)

            batch_node_embedding[batch_index, :, :] = sample_node_embedding
            batch_edge_embedding[batch_index, :, :] = sample_edge_embedding

        node_logits = self.classifier_node(batch_node_embedding)
        edge_logits = self.classifier_edge(batch_edge_embedding)

        outputs = (logits, node_logits, edge_logits) + outputs[2:]
        if qa_labels is not None:
            loss_fct = CrossEntropyLoss()
            qa_loss = loss_fct(logits.view(-1, self.num_labels), qa_labels.view(-1))
            node_loss = loss_fct(
                node_logits.view(-1, self.num_labels), node_labels.view(-1)
            )
            # print(edge_logits.view(-1, self.num_labels_edge))
            # print(edge_labels.view(-1))
            edge_loss = loss_fct(
                edge_logits.view(-1, self.num_labels_edge), edge_labels.view(-1)
            )
            total_loss = qa_loss + node_loss + edge_loss
            outputs = (total_loss, qa_loss, node_loss, edge_loss) + outputs

        return outputs

    def predict(self, triples, rules, question, device: str = "cpu"):
        with torch.no_grad():
            context_tokens = []
            proof_offset = []
            sentences = ["<s>"]

            nfact = len(triples)
            nrule = len(rules)
            node_length = nfact + nrule + 1
            edge_length = node_length**2

            for s in triples.values():
                sentences.append(s)
            for s in rules.values():
                sentences.append(s)
            for s in sentences:
                sentence_tokens = self.proofwriter_collator.tokenizer.tokenize(s)
                context_tokens.extend(sentence_tokens)
                proof_offset.append(len(context_tokens))
            sentences.append("</s>")
            sentences.append("</s>")
            sentences.append(question)
            sentences.append("</s>")
            context = "".join(sentences)
            proofs_offsets = torch.tensor([proof_offset])

            tokenized_context = self.proofwriter_collator.tokenizer(
                [context], add_special_tokens=False, padding=True, return_tensors="pt"
            )
            logits = self(
                tokenized_context.to(device),
                proofs_offsets.to(device),
                max_node_length=node_length,
                max_edge_length=edge_length,
                device=device,
            )
            pred_qa_label = logits[0].argmax()
            return pred_qa_label.item()
