import traceback

from vero.metrics import MetricBase
import gc
from .alignscore import AlignScorer
import torch
# from tracing_components import logger


# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

class AlignScore(MetricBase):
    '''
        Calculates Align Score which is the measure of faithfulness.

        Methods
        ---------
        1.  __init__()
            Initializes the model for metric.

        2. evaluate(reference,candidate) -> float
            Returns the align score.

        :returns: float
    '''
    name = 'align_score'

    def __init__(self):
        self.scorer = AlignScorer(model='roberta-large', batch_size=32, device='cuda',
                            ckpt_path=r'C:\Users\HP\PycharmProjects\PythonProject2\vero\metrics\align_score\alignscore\alignscore_hf_model',
                            evaluation_mode='nli_sp')

    def __enter__(self):
        return self

    # TODO: download model locally so that no connection error appears
    def evaluate(self,reference: str | list, candidate: str | list) -> float | None:
        '''
        :param reference: Pass the chunks for reference.
        :param candidate: Pass the answer that has to be checked.

        :return: float
        '''
        scorer = self.scorer
        try:
            # logger.info('Starting AlignScore calculation')
            if isinstance(reference, str):
                score = scorer.score(contexts=[reference], claims=[candidate])
                return round(score[0], 2)
            elif isinstance(reference, list):
                score, avg_score = 0, 0
                if isinstance(candidate, str):
                    candidate = [candidate]
                for doc in reference:
                    score += scorer.score(contexts=[doc], claims=candidate)[0]
                    avg_score = score / len(reference)
                return round(avg_score, 2)

        except Exception as e:
            # logger.error('AlignScore calculation failed\nError:', e)
            print('AlignScore calculation failed\nError:', traceback.format_exc())
            return None


    def __exit__(self, exc_type, exc_value, traceback):
        del self.scorer
        self.scorer = None
        gc.collect()
        torch.cuda.empty_cache()