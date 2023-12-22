import logging
import sys
from typing import Dict, Tuple

from allennlp_models import pretrained

logging.disable(sys.maxsize)


class CoreferenceResolution:
    def __init__(self) -> None:
        self.model = pretrained.load_predictor("coref-spanbert")

    def coref_resolved(self, passage: str) -> str:
        """
        Resolves coreferences in the given passage.

        Args:
            passage (str): The input passage containing coreferences.

        Returns:
            str: The passage with resolved coreferences.
        """
        return self.model.coref_resolved(passage)


if __name__ == "__main__":
    analyzer = CoreferenceResolution()
    print(
        analyzer.coref_resolved(
            "Zidane played football. He was one of the greatest footballer."
        )
    )
