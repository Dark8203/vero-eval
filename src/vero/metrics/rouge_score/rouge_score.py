import traceback

from vero.metrics import MetricBase
import gc
from rouge_score import rouge_scorer
import torch
# from tracing_components import logger


# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

class RougeScore(MetricBase):
    '''
        Calculates ROUGEL Score which helps in measuring recall (focuses on the Longest Common Subsequence (LCS)).

        Methods
        ---------
        1.  __init__()
            Initializes the model for metric.

        2. evaluate(reference,candidate) -> tuple
            Returns the rouge score tuple which contains precision, recall and f1 score.

        :returns: tuple
        '''
    name = 'rouge_score'

    def __init__(self):
        self.rg_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def __enter__(self):
        return self

    def evaluate(self,reference: str | list, candidate: str | list) -> tuple | None:
        '''
        :param reference: Pass the chunks for reference.
        :param candidate: Pass the answer that has to be checked.

        :return: tuple
        '''
        # logger.info('Starting ROUGE Score calculation')
        rg_scorer = self.rg_scorer
        try:
            if isinstance(reference, str):
                score = rg_scorer.score(reference, candidate)
                return (round(score['rougeL'].precision, 2), round(score['rougeL'].recall, 2),
                        round(score['rougeL'].fmeasure, 2))
            elif isinstance(reference, list):
                sum_p, sum_r, sum_f1 = 0, 0, 0
                if isinstance(candidate, list):
                    candidate = candidate[0]
                for doc in reference:
                    score = rg_scorer.score(doc, candidate)
                    sum_p += score['rougeL'].precision
                    sum_r += score['rougeL'].recall
                    sum_f1 += score['rougeL'].fmeasure
                avg_p = sum_p / len(reference)
                avg_f1 = sum_f1 / len(reference)
                avg_r = sum_r / len(reference)
                return (round(avg_p, 2), round(avg_r, 2), round(avg_f1, 2))

        except Exception as e:
            # logger.info('ROUGE score calculation failed\nError:', e)
            print('Error calculating ROUGE Score\nError:', traceback.format_exc())
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        del self.rg_scorer
        self.rg_scorer = None
        gc.collect()
        torch.cuda.empty_cache()