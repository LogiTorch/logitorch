from typing import Optional

from textblob import TextBlob

from utilities.exceptions import OutOfRangeError


class SentimentAnalysisTextBlob:
    def __init__(self) -> None:
        self.textblob_analyzer = TextBlob

    def sentiment(self, passage: str, threshold_polarity: float = 0.0) -> Optional[str]:
        """
        Analyzes the sentiment of a given passage using TextBlob.

        Args:
            passage (str): The text passage to analyze.
            threshold_polarity (float, optional): The threshold polarity value. Defaults to 0.0.

        Returns:
            Optional[str]: The sentiment of the passage, either "Positive" or "Negative", or None if an error occurs.
        """
        try:
            if threshold_polarity < -1.0 or threshold_polarity > 1.0:
                raise OutOfRangeError(-1.0, 1.0)

            polarity = self.textblob_analyzer(passage).sentiment.polarity
            if polarity > threshold_polarity:
                return "Positive"
            return "Negative"
        except OutOfRangeError as err:
            print(err.message)
        return None
