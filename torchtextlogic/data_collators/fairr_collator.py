from transformers import RobertaTokenizer


class RuleSelectionProofWriterIterCollator:
    def __init__(self, pretrained_roberta_tokenizer: str) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_roberta_tokenizer)

    def __call__(self, batch):
        batch_x = []
        batch_y = []
        y_indices = []
        for facts_list, rules_list, question, _, proofs in batch:
            rules = []
            facts = []
            y = 0
            for fact in facts_list.values():
                facts.append(fact)

            for cnt, rule in enumerate(rules_list):
                rules.append(rules_list[rule])
                if proofs[0] is not None:
                    if rule in proofs[0]:
                        y = cnt + 1

            batch_x.append(question + " </s> " + " ".join(facts) + " </s> ".join(rules))
            batch_y.append(y)

        batch_x = self.tokenizer(batch_x, padding=True, return_tensors="pt")
        id_sep_token = 2
        print(batch_x["input_ids"])
        print(id_sep_token)
        return batch_x, batch_y
