import re
from typing import List

EXCPLICIT_CONNECTIVES = {
    "once",
    "although",
    "though",
    "but",
    "because",
    "nevertheless",
    "before",
    "for example",
    "until",
    "if",
    "previously",
    "when",
    "and",
    "so",
    "then",
    "while",
    "as long as",
    "however",
    "also",
    "after",
    "separately",
    "still",
    "so that",
    "or",
    "moreover",
    "in addition",
    "instead",
    "on the other hand",
    "as",
    "for instance",
    "nonetheless",
    "unless",
    "meanwhile",
    "yet",
    "since",
    "rather",
    "in fact",
    "indeed",
    "later",
    "ultimately",
    "as a result",
    "either or",
    "therefore",
    "in turn",
    "thus",
    "in particular",
    "further",
    "afterward",
    "next",
    "similarly",
    "besides",
    "if and when",
    "nor",
    "alternatively",
    "whereas",
    "overall",
    "by comparison",
    "till",
    "in contrast",
    "finally",
    "otherwise",
    "as if",
    "thereby",
    "now that",
    "before and after",
    "additionally",
    "meantime",
    "by contrast",
    "if then",
    "likewise",
    "in the end",
    "regardless",
    "thereafter",
    "earlier",
    "in other words",
    "as soon as",
    "except",
    "in short",
    "neither nor",
    "furthermore",
    "lest",
    "as though",
    "specifically",
    "conversely",
    "consequently",
    "as well",
    "much as",
    "plus",
    "and",
    "hence",
    "by then",
    "accordingly",
    "on the contrary",
    "simultaneously",
    "for",
    "in sum",
    "when and if",
    "insofar as",
    "else",
    "as an alternative",
    "on the one hand on the other hand",
}

PUNCTUATION_DELIMETERS = {".", ",", ";", ":"}


class DiscourseDelimiter:
    def __init__(self) -> None:
        self.regex_explicit_connectives = "|".join(
            rf"\b{conn}\b" for conn in EXCPLICIT_CONNECTIVES
        )
        self.regex_punctuation_delimiters = "|".join(
            rf"\{punct}" for punct in PUNCTUATION_DELIMETERS
        )

    def split_edu(self, passage: str) -> str:
        edu_explicit_connectives = "<CONNECTIVE>".join(
            edu for edu in self.split_explicit_connectives(passage)
        )
        edus = " <PUNCT>".join(
            edu for edu in self.split_punctuation_delimiters(edu_explicit_connectives)
        )
        return edus

    def split_explicit_connectives(self, passage: str) -> List[str]:
        split_passage = re.split(self.regex_explicit_connectives, passage)
        split_passage = [x for x in split_passage if x.strip()]
        return split_passage

    def split_punctuation_delimiters(self, passage: str) -> List[str]:
        split_passage = re.split(self.regex_punctuation_delimiters, passage)
        split_passage = [x for x in split_passage if x.strip()]
        return split_passage


if __name__ == "__main__":
    d = DiscourseDelimiter()
    passage = "Digital systems. are the best information systems because  error cannot occur in the emission of digital signals."
    print(d.split_edu(passage))
