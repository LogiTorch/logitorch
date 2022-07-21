from transformers import T5Tokenizer


class ProofWriterQACollator:
    def __init__(self, pretrained_t5_tokenizer: str) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_tokenizer)

    def __call__(self, batch):
        contexts = []
        questions = []
        labels = []

        for i in batch:
            sentences = []
            for s in i[0].values():
                sentences.append(s)
            for s in i[1].values():
                sentences.append(s)

            contexts.append("".join(sentences))
            questions.append(i[2])
            labels.append(str(i[3]))

        batch_x = self.tokenizer(
            contexts,
            questions,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        batch_y = self.tokenizer(
            labels, padding=True, max_length=512, truncation=True, return_tensors="pt"
        )

        return batch_x, batch_y.input_ids


class ProofWriterProofGenerationAllCollator:
    def __init__(self, pretrained_t5_tokenizer: str) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_tokenizer)

    def __call__(self, batch):
        contexts = []
        questions = []
        labels = []
        proofs = []

        for i in batch:
            sentences = []
            for s in i[0].values():
                sentences.append(s)
            for s in i[1].values():
                sentences.append(s)

            contexts.append("".join(sentences))
            questions.append(i[2])
            labels.append(str(i[3]))
            proofs.append(i[4])
        # print(proofs)
        batch_x = self.tokenizer(
            contexts,
            questions,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        batch_y = self.tokenizer(
            labels,
            proofs,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        return batch_x, batch_y.input_ids
