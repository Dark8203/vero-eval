from vero.metrics import MetricBase
import gc
from .bartscore import BARTScorer
import torch
# from tracing_components import logger


# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

class BartScore(MetricBase):
    '''
        Calculates Bart Score which helps in comparing two different models/configurations.
        Doesn't have significance on its own, it's a comparison score.

        Methods
        ---------
        1.  __init__()
            Initializes the model for metric.

        2. evaluate(reference,candidate,batch_size) -> float
            Returns the bart score.

        :returns: float
    '''
    name = 'bart_score'

    def __init__(self):
        self.bart_scorer = BARTScorer()

    def __enter__(self):
        return self

    def evaluate(self,reference: str | list, candidate: str | list, batch_size: int = 4) -> float | None:
        '''
        :param reference: Pass the chunks for reference.
        :param candidate: Pass the answer that has to be checked.
        :param batch_size: Pass the number of samples to evaluate. (Optional)

        :returns: float
        '''

        # logger.info('Starting Bart Score calculation')
        bart_scorer = self.bart_scorer
        score = 0
        try:
            if isinstance(reference, str):
                score = bart_scorer.score([reference], [candidate], batch_size=batch_size)
                return round(score[0], 2)
            elif isinstance(reference, list):
                sum_score = 0
                if isinstance(candidate, str):
                    candidate = [candidate]
                for doc in reference:
                    score = bart_scorer.score([doc], candidate, batch_size=batch_size)
                    sum_score += score[0]
                avg_score = sum_score / len(reference)
                return round(avg_score, 2)
        except Exception as e:
            # logger.error('Error calculating Bart Score\nError:', e)
            return None
        # score = scorer.multi_ref_score([ref],[candidate],batch_size = 6)


    def __exit__(self, exc_type, exc_value, traceback):
        del self.bart_scorer
        self.bart_scorer = None
        gc.collect()
        torch.cuda.empty_cache()