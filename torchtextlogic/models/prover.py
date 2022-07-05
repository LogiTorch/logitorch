import torch.nn as nn
import torch
from torch.nn.init import xavier_normal
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead


class _NodeClassificationHead(nn.Module):
    def __init__(self, config):
        super(self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class _EdgeClassificationHead(nn.Module):
    def __init__(self, config):
        super(self).__init__()
        self.dense = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

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
        self.num_labels = 2
        self.encoder = RobertaModel.from_pretrained(pretrained_roberta_model)
        self.config = self.encoder.config
        self.naf_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.classifier = RobertaClassificationHead(self.config)
        self.classifier_node = _NodeClassificationHead(self.config)
        self.classifier_edge = _EdgeClassificationHead(self.config)

        xavier_normal(self.naf_layer)
        # xavier_normal(self.classifier_node)
        # xavier_normal(self.classifier_edge)

    def forward(self, x, y=None):
        outputs = self.encoder(x)
        sequence_outputs = outputs[0]
        cls_outputs = sequence_outputs[:, 0, :]
        naf_outputs = self.naf_layer(cls_outputs)
        logits = self.classifier(sequence_outputs)


# PRover("roberta-base")

x = torch.tensor([1, 2, 3])
print(x.repeat(4, 2, 1))


class Node:
    def __init__(self, head):
        self.head = head

    def __str__(self):
        return str(self.head)


def get_proof_graph(proof_str):
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


print(get_proof_graph("[(((triple1 triple11) -> rule2))"))
