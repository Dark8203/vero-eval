from vero.metrics import MetricBase
import gc
from bert_score import BERTScorer
import torch
import traceback
# from tracing_components import logger


# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

class BertScore(MetricBase):
    '''
        Calculates Bert Score which compares tokens of candidate and reference texts via contextual embeddings (like BERT), computing precision, recall, and F1 to assess semantic similarity beyond exact word overlap.

        Methods
        ---------
        1.  __init__()
            Initializes the model for metric.

        2. evaluate(reference,candidate) -> tuple
            Returns the bert score tuple which contains precision, recall and f1 score.

        :returns: tuple
        '''
    name = 'bert_score'

    def __init__(self):
        self.bert_scorer = BERTScorer(model_type='bert-base-uncased', device=device)

    def __enter__(self):
        return self

    def evaluate(self,reference: str | list, candidate: str | list) -> tuple | None:
        '''
        :param reference: Pass the chunks for reference.
        :param candidate: Pass the answer that has to be checked.

        :return: tuple
        '''
        # logger.info('Starting BERTscore calculation')
        bert_scorer = self.bert_scorer
        try:
            if isinstance(reference, str):
                p, r, f1 = bert_scorer.score([candidate], [reference])
                return (round(p.item(), 2), round(r.item(), 2), round(f1.item(), 2))
            elif isinstance(reference, list):
                sum_p, sum_r, sum_f1 = 0, 0, 0
                if isinstance(candidate, str):
                    candidate = [candidate]
                for doc in reference:
                    p, r, f1 = bert_scorer.score(candidate, [doc])
                    sum_p += p.item()
                    sum_r += r.item()
                    sum_f1 += f1.item()
                avg_p = sum_p / len(reference)
                avg_f1 = sum_f1 / len(reference)
                avg_r = sum_r / len(reference)
                return (round(avg_p, 2), round(avg_r, 2), round(avg_f1, 2))
        except Exception as e:
            print('Error calculating Bert Score\nError:', traceback.format_exc())
            # logger.info('BERTscore calculation failed\nError:', e)
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        del self.bert_scorer
        self.bert_scorer = None
        gc.collect()
        torch.cuda.empty_cache()