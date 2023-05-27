from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer

from logitorch.datasets.proof_qa.proofwriter_dataset import PROOFWRITER_LABEL_TO_ID


class Node:
    def __init__(self, head: str) -> None:
        self.head = head

    def __str__(self) -> str:
        return str(self.head)


class PRoverProofWriterCollator:
    def __init__(self, pretrained_roberta_tokenizer: str) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_roberta_tokenizer)

    def get_proof_graph_with_fail(self, proof_str: str) -> Tuple[List[str], List[str]]:
        proof_str = proof_str[:-2].split("=")[1].strip()[1:-1]
        nodes = proof_str.split(" <- ")

        all_nodes = []
        all_edges = []
        for i in range(len(nodes) - 1):
            all_nodes.append(nodes[i])
            if nodes[i + 1] != "FAIL":
                all_edges.append((nodes[i + 1], nodes[i]))

        return all_nodes, all_edges

    def get_proof_graph(
        self, proof_str: str
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        stack = []
        last_open = 0
        last_open_index = 0
        pop_list = []
        all_edges = []
        all_nodes = []

        proof_str = proof_str.replace("(", " ( ")
        proof_str = proof_str.replace(")", " ) ")
        proof_str = proof_str.split()

        should_join = False
        for i in range(len(proof_str)):

            _s = proof_str[i]
            x = _s.strip()
            if len(x) == 0:
                continue

            if x == "(":
                stack.append((x, i))
                last_open = len(stack) - 1
                last_open_index = i
            elif x == ")":
                for j in range(last_open + 1, len(stack)):
                    if isinstance(stack[j][0], Node):
                        pop_list.append((stack[j][1], stack[j][0]))

                stack = stack[:last_open]
                for j in range((len(stack))):
                    if stack[j][0] == "(":
                        last_open = j
                        last_open_index = stack[j][1]

            elif x == "[" or x == "]":
                pass
            elif x == "->":
                should_join = True
            else:
                # terminal
                if x not in all_nodes:
                    all_nodes.append(x)

                if should_join:

                    new_pop_list = []
                    # Choose which ones to add the node to
                    for (index, p) in pop_list:
                        if index < last_open_index:
                            new_pop_list.append((index, p))
                        else:
                            all_edges.append((p.head, x))
                    pop_list = new_pop_list

                stack.append((Node(x), i))

                should_join = False

        return all_nodes, all_edges

    def get_node_edge_label_constrained(
        self, x: str
    ) -> Tuple[List[int], List[np.ndarray]]:

        proofs = x[4]
        nrule = len(x[1])
        sentence_scramble = [i[0] + 1 for i in enumerate(x[0])]
        nfact = len(sentence_scramble)
        sentence_scramble += [nfact + 1 + i[0] for i in enumerate(x[1])]

        proof = proofs.split("OR")[0]
        node_label = [0] * (nfact + nrule + 1)
        edge_label = np.zeros((nfact + nrule + 1, nfact + nrule + 1), dtype=int)

        if "FAIL" in proof:
            nodes, edges = self.get_proof_graph_with_fail(proof)
        else:
            nodes, edges = self.get_proof_graph(proof)

        component_index_map = {}
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
            else:
                component = "rule" + str(index - nfact)
            component_index_map[component] = i
        component_index_map["NAF"] = nfact + nrule

        for node in nodes:
            index = component_index_map[node]
            node_label[index] = 1

        edges = list(set(edges))
        for edge in edges:
            start_index = component_index_map[edge[0]]
            end_index = component_index_map[edge[1]]
            edge_label[start_index][end_index] = 1

        # Mask impossible edges
        for i in range(len(edge_label)):
            for j in range(len(edge_label)):
                # Ignore diagonal
                if i == j:
                    edge_label[i][j] = -100
                    continue

                # Ignore edges between non-nodes
                if node_label[i] == 0 or node_label[j] == 0:
                    edge_label[i][j] = -100
                    continue

                is_fact_start = False
                is_fact_end = False
                if i == len(edge_label) - 1 or sentence_scramble[i] <= nfact:
                    is_fact_start = True
                if j == len(edge_label) - 1 or sentence_scramble[j] <= nfact:
                    is_fact_end = True

                # No edge between fact/NAF -> fact/NAF
                if is_fact_start and is_fact_end:
                    edge_label[i][j] = -100
                    continue

                # No edge between Rule -> fact/NAF
                if not is_fact_start and is_fact_end:
                    edge_label[i][j] = -100
                    continue

        return node_label, list(edge_label.flatten())

    def __call__(
        self, batch
    ) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        contexts = []
        proofs_offsets = []
        node_labels = []
        edge_labels = []
        labels = []
        for i in batch:
            context_tokens = []
            proof_offset = []
            sentences = ["<s>"]
            for s in i[0].values():
                sentences.append(s)
            for s in i[1].values():
                sentences.append(s)
            for s in sentences:
                sentence_tokens = self.tokenizer.tokenize(s)
                context_tokens.extend(sentence_tokens)
                proof_offset.append(len(context_tokens))
            sentences.append("</s>")
            sentences.append("</s>")
            sentences.append(i[2])
            sentences.append("</s>")
            contexts.append("".join(sentences))
            proofs_offsets.append(torch.tensor(proof_offset))
            node_label, edge_label = self.get_node_edge_label_constrained(i)
            node_labels.append(torch.tensor(node_label))
            edge_labels.append(torch.LongTensor(edge_label))
            labels.append(PROOFWRITER_LABEL_TO_ID[str(i[3])])

        tokenized_batch = self.tokenizer(
            contexts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        proofs_offsets = pad_sequence(proofs_offsets, batch_first=True)
        node_labels = pad_sequence(node_labels, batch_first=True, padding_value=-100)
        edge_labels = pad_sequence(edge_labels, batch_first=True, padding_value=-100)
        labels = torch.tensor(labels)

        return tokenized_batch, proofs_offsets, node_labels, edge_labels, labels
