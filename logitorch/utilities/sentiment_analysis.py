from typing import Optional

from textblob import TextBlob

from utilities.exceptions import OutOfRangeError


class SentimentAnalysisTextBlob:
    def __init__(self) -> None:
        self.textblob_analyzer = TextBlob

    def sentiment(self, passage: str, threshold_polarity: float = 0.0) -> Optional[str]:
        """_summary_

        :param passage: _description_
        :type passage: str
        :param threshold_polarity: _description_, defaults to 0.0
        :type threshold_polarity: float, optional
        :return: _description_
        :rtype: str
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
