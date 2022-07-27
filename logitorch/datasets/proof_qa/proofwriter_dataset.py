import os
import re
from typing import Dict, List, Optional, Tuple, Union

from logitorch.datasets.base import AbstractProofQADataset
from logitorch.datasets.exceptions import (
    AbductionClosedWorldAssumptionError,
    DatasetNameError,
    SplitSetError,
    TaskError,
)
from logitorch.datasets.utils import (
    DATASETS_FOLDER,
    SPLIT_SETS,
    download_dataset,
    read_jsonl,
)

PROOFWRITER_DATASET_ZIP_URL = (
    "https://www.dropbox.com/s/5dpm2yjcm5h9hsd/proofwriter_dataset.zip?dl=1"
)
PROOFWRITER_DATASET = "proofwriter_dataset"
PROOFWRITER_SUB_DATASETS = [
    "birds-electricity",
    "depth-0",
    "depth-1",
    "depth-2",
    "depth-3",
    "depth-3ext",
    "depth-3ext-NatLang",
    "depth-5",
    "NatLang",
]
PROOFWRITER_TASKS = [
    "proof_generation_all",
    "proof_generation_iter",
    "implication_enumeration",
    "abduction",
]
PROOFWRITER_DATASET_FOLDER = f"{DATASETS_FOLDER}/{PROOFWRITER_DATASET}"
PROOFWRITER_LABEL_TO_ID = {"False": 0, "True": 1, "Unknown": 2}
PROOFWRITER_ID_TO_LABEL = {0: "False", 1: "True", 2: "Unknown"}


class ProofWriterDataset(AbstractProofQADataset):
    def __init__(
        self,
        dataset_name: str,
        split_set: str,
        task: str,
        open_world_assumption: bool = False,
    ) -> None:
        try:
            if dataset_name not in PROOFWRITER_SUB_DATASETS:
                raise DatasetNameError()
            if split_set != "test" and dataset_name == "birds-electricity":
                raise SplitSetError(["test"])

            if split_set == "val":
                split_set = "dev"
            elif split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if task not in PROOFWRITER_TASKS:
                raise TaskError()

            if not os.path.exists(PROOFWRITER_DATASET_FOLDER):
                download_dataset(PROOFWRITER_DATASET_ZIP_URL, PROOFWRITER_DATASET)

            self.dataset_name = dataset_name
            self.split_set = split_set
            self.task = task
            self.world_assumption = "CWA"

            if open_world_assumption:
                self.world_assumption = "OWA"

            if self.world_assumption == "CWA" and self.task == "abduction":
                raise AbductionClosedWorldAssumptionError()

            if self.task == "proof_generation_all":
                self.dataset_path = f"{PROOFWRITER_DATASET_FOLDER}/{self.world_assumption}/{self.dataset_name}/meta-{self.split_set}.jsonl"
                (
                    self.triples,
                    self.rules,
                    self.questions,
                    self.labels,
                    self.proofs,
                    self.proofs_intermerdiates,
                    self.depths,
                ) = self.__read_dataset_proof_generation_all(
                    "triples", "rules", "questions"
                )
            elif self.task == "proof_generation_iter":
                self.dataset_path = f"{PROOFWRITER_DATASET_FOLDER}/{self.world_assumption}/{self.dataset_name}/meta-stage-{self.split_set}.jsonl"
                (
                    self.triples,
                    self.rules,
                    self.labels,
                    self.proofs,
                ) = self.__read_dataset_proof_generation_iter(
                    "triples", "rules", "allInferences"
                )
            elif self.task == "implication_enumeration":
                self.dataset_path = f"{PROOFWRITER_DATASET_FOLDER}/{self.world_assumption}/{self.dataset_name}/meta-{self.split_set}.jsonl"
                (
                    self.triples,
                    self.rules,
                    self.labels,
                ) = self.__read_dataset_implication_enumeration(
                    "triples", "rules", "proofDetails"
                )
            elif self.task == "abduction":
                self.dataset_path = f"{PROOFWRITER_DATASET_FOLDER}/{self.world_assumption}/{self.dataset_name}/meta-abduct-{self.split_set}.jsonl"
                (
                    self.triples,
                    self.rules,
                    self.questions,
                    self.labels,
                    self.proofs,
                ) = self.__read_dataset_abduction("triples", "rules", "abductions")

        except DatasetNameError as err:
            print(err.message)
            print(f"The ProofWriter datasets are: {PROOFWRITER_SUB_DATASETS}")
        except SplitSetError as err:
            print(err.message)
        except TaskError as err:
            print(err.message)
            print(f"The ProofWriter tasks are: {PROOFWRITER_TASKS}")
        except AbductionClosedWorldAssumptionError as err:
            print(err.message)

    def __read_dataset_proof_generation_all(
        self,
        triples_key: str,
        rules_key: str,
        questions_key: str,
    ) -> Tuple[
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[List[str]],
        List[List[str]],
        List[List[str]],
        List[List[str]],
        List[List[int]],
    ]:
        data = read_jsonl(self.dataset_path)
        triples_list = []
        rules_list = []
        questions_list = []
        labels_list = []
        proofs_list = []
        proofs_intermediates_list = []
        depths_list = []

        proofs_intermediates_key = "proofsWithIntermediates"
        for i in data:
            triples = {}
            rules = {}
            questions = []
            proofs = []
            proofs_intermediates = []
            labels = []
            depths = []

            for t, val in i[triples_key].items():
                triples[t] = val["text"]

            for r, val in i[rules_key].items():
                rules[r] = val["text"]

            for q in i[questions_key].values():
                questions.append(q["question"])
                if proofs_intermediates_key in q:
                    tmp_proof = []
                    for p in q[proofs_intermediates_key]:
                        str_proof = f"{p['representation']}"
                        if len(p["intermediates"]) > 0:
                            str_proof += " ; "
                            str_proof += "with "
                            for intr, val in p["intermediates"].items():
                                str_proof += f"{intr} = {val['text']}"
                        tmp_proof.append(str_proof)

                labels.append(q["answer"])
                proofs.append(q["proofs"])
                proofs_intermediates.append(tmp_proof)
                depths.append(q["QDep"])

            for q, l, p, p_i, d in zip(
                questions, labels, proofs, proofs_intermediates, depths
            ):
                triples_list.append(triples)
                rules_list.append(rules)
                questions_list.append(q)
                labels_list.append(l)
                proofs_list.append(p)
                proofs_intermediates_list.append(p_i)
                depths_list.append(d)

        # print(
        #     triples_list[0],
        #     rules_list[0],
        #     questions_list[0],
        #     labels_list[0],
        #     proofs_list[0],
        # )
        return (
            triples_list,
            rules_list,
            questions_list,
            labels_list,
            proofs_list,
            proofs_intermediates_list,
            depths_list,
        )

    def __read_dataset_proof_generation_iter(
        self, triples_key: str, rules_key: str, proofs_key: str
    ) -> Tuple[
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[List[str]],
        List[List[str]],
    ]:
        data = read_jsonl(self.dataset_path)
        triples_list = []
        rules_list = []
        labels_list = []
        proofs_list = []

        for i in data:
            triples = {}
            rules = {}
            inferences = []
            proofs = []

            for t, val in i[triples_key].items():
                triples[t] = val["text"]
            for r, val in i[rules_key].items():
                rules[r] = val["text"]
            for val in i[proofs_key]:
                inferences.append(val["text"])
                proofs.append(val["proofs"])

            triples_list.append(triples)
            rules_list.append(rules)

            if len(inferences) > 0:
                labels_list.append(inferences)
                proofs_list.append(proofs)
            else:
                labels_list.append([None])
                proofs_list.append([None])

        # print(triples_list[30], rules_list[30], labels_list[30], proofs_list[30])
        return triples_list, rules_list, labels_list, proofs_list

    def __read_dataset_implication_enumeration(
        self, triples_key: str, rules_key: str, labels_key: str
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[List[str]]]:
        data = read_jsonl(self.dataset_path)
        triples_list = []
        rules_list = []
        labels_list = []

        for i in data:
            triples = {}
            rules = {}
            labels = []

            for t, val in i[triples_key].items():
                triples[t] = val["text"]

            for r, val in i[rules_key].items():
                rules[r] = val["text"]

            for p in i[labels_key]:
                labels.append(p["text"])

            triples_list.append(triples)
            rules_list.append(rules)

            if len(labels) > 0:
                labels_list.append(labels)
            else:
                labels_list.append([None])

        print(triples_list[30], rules_list[30], labels_list[30])

        return triples_list, rules_list, labels_list

    def __read_dataset_abduction(
        self, triples_key: str, rules_key: str, abductions_key: str
    ) -> Tuple[
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[List[str]],
        List[List[str]],
        List[List[str]],
    ]:
        data = read_jsonl(self.dataset_path)
        triples_list = []
        rules_list = []
        questions_list = []
        labels_list = []
        proofs_list = []

        for i in data:
            triples = {}
            rules = {}
            questions = []
            labels = []
            proofs = []

            for t, val in i[triples_key].items():
                triples[t] = val["text"]

            for r, val in i[rules_key].items():
                rules[r] = val["text"]

            for abduc in i[abductions_key].values():
                questions.append(abduc["question"])
                tmp_labels = []
                tmp_proofs = []
                if len(abduc["answers"]) > 0:
                    for answer in abduc["answers"]:
                        tmp_labels.append(answer["text"])
                        tmp_proofs.append(answer["proof"])
                else:
                    tmp_labels.append(None)
                    tmp_proofs.append(None)

                labels.append(tmp_labels)
                proofs.append(tmp_proofs)

            for q, l, p in zip(questions, labels, proofs):
                triples_list.append(triples)
                rules_list.append(rules)
                questions_list.append(q)
                labels_list.append(l)
                proofs_list.append(p)

        # print(triples_list[2150], rules_list[1250], questions_list[1250], labels_list[1250], proofs_list[1250])
        return triples_list, rules_list, questions_list, labels_list, proofs_list

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[
            Dict[str, str],
            Dict[str, str],
            List[str],
            List[str],
            List[str],
            List[str],
            List[int],
        ],
        Tuple[Dict[str, str], Dict[str, str], List[str], List[str], List[str]],
        Tuple[Dict[str, str], Dict[str, str], List[str], List[str]],
        Tuple[Dict[str, str], Dict[str, str], List[str]],
    ]:
        if self.task == "proof_generation_all":
            return (
                self.triples[index],
                self.rules[index],
                self.questions[index],
                self.labels[index],
                self.proofs[index],
                self.proofs_intermerdiates[index],
                self.depths[index],
            )
        elif self.task == "abduction":
            return (
                self.triples[index],
                self.rules[index],
                self.questions[index],
                self.labels[index],
                self.proofs[index],
            )
        elif self.task == "proof_generation_iter":
            return (
                self.triples[index],
                self.rules[index],
                self.labels[index],
                self.proofs[index],
            )
        else:
            return self.triples[index], self.rules[index], self.labels[index]

    def __str__(self) -> str:
        return f'The {self.split_set} set of {self.dataset_name}\'s ProofWriter for the task of "{self.task}" has {self.__len__()} instances'

    def __len__(self) -> int:
        return len(self.triples)


class FaiRRProofWriterDataset:
    def __init__(
        self,
        dataset_name: str,
        split_set: str,
        open_world_assumption: bool = False,
    ) -> None:
        try:
            if dataset_name not in PROOFWRITER_SUB_DATASETS:
                raise DatasetNameError()
            if split_set != "test" and dataset_name == "birds-electricity":
                raise SplitSetError(["test"])

            if split_set == "val":
                split_set = "dev"
            elif split_set not in SPLIT_SETS:
                raise SplitSetError(SPLIT_SETS)

            if not os.path.exists(PROOFWRITER_DATASET_FOLDER):
                download_dataset(PROOFWRITER_DATASET_ZIP_URL, PROOFWRITER_DATASET)

            self.dataset_name = dataset_name
            self.split_set = split_set
            self.world_assumption = "CWA"

            if open_world_assumption:
                self.world_assumption = "OWA"

            self.dataset_path = f"{PROOFWRITER_DATASET_FOLDER}/{self.world_assumption}/{self.dataset_name}/meta-stage-{self.split_set}.jsonl"
            self.no_staged_dataset_path = f"{PROOFWRITER_DATASET_FOLDER}/{self.world_assumption}/{self.dataset_name}/meta-{self.split_set}.jsonl"
            (
                self.triples,
                self.rules,
                self.questions,
                self.labels,
                self.proofs,
            ) = self.__read_dataset_proof_generation_iter(
                "triples", "rules", "allInferences"
            )

        except DatasetNameError as err:
            print(err.message)
            print(f"The ProofWriter datasets are: {PROOFWRITER_SUB_DATASETS}")
        except SplitSetError as err:
            print(err.message)

    def __read_dataset_proof_generation_iter(
        self, triples_key: str, rules_key: str, proofs_key: str
    ) -> Tuple[
        List[Dict[str, str]],
        List[Dict[str, str]],
        List[List[str]],
        List[List[str]],
    ]:
        data = read_jsonl(self.dataset_path)
        no_staged_data = read_jsonl(self.no_staged_dataset_path)
        questions = {}

        for i in no_staged_data:
            questions_tmp = []
            for q in i["questions"].values():
                questions_tmp.append(q["question"])
            questions[i["id"]] = questions_tmp

        triples_list = []
        rules_list = []
        questions_list = []
        labels_list = []
        proofs_list = []

        for i in data:

            id = re.sub("-add[0-9]+", "", i["id"])

            for q in questions[id]:
                triples = {}
                rules = {}
                inferences = []
                proofs = []

                for t, val in i[triples_key].items():
                    triples[t] = val["text"]
                for r, val in i[rules_key].items():
                    rules[r] = val["text"]
                for val in i[proofs_key]:
                    inferences.append(val["text"])
                    proofs.append(val["proofs"])

                triples_list.append(triples)
                rules_list.append(rules)
                questions_list.append(q)

                if len(inferences) > 0:
                    labels_list.append(inferences)
                    proofs_list.append(proofs)
                else:
                    labels_list.append([None])
                    proofs_list.append([None])

        return triples_list, rules_list, questions_list, labels_list, proofs_list

    def __getitem__(self, index: int):

        return (
            self.triples[index],
            self.rules[index],
            self.questions[index],
            self.labels[index],
            self.proofs[index],
        )

    def __str__(self) -> str:
        return f'The {self.split_set} set of {self.dataset_name}\'s ProofWriter for the task of "{self.task}" has {self.__len__()} instances'

    def __len__(self) -> int:
        return len(self.triples)
