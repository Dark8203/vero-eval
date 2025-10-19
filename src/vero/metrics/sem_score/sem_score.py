from vero.metrics import MetricBase
import gc
from .semscore import EmbeddingModelWrapper
import torch
# from tracing_components import logger


# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

class SemScore(MetricBase):
    '''
        Calculates SEMScore which evaluates outputs by computing the semantic textual similarity between model output and gold reference using embedding.

        Methods
        ---------
        1.  __init__()
            Initializes the model for metric.

        2. evaluate(reference,candidate) -> tuple
            Returns the semscore.

        :returns: float
        '''
    name = 'sem_score'

    def __init__(self):
        self.em = EmbeddingModelWrapper()

    def __enter__(self):
        return self

    def evaluate(self,reference: str | list, candidate: str | list) -> float | None:
        '''
        :param reference: Pass the reference chunks.
        :param candidate: Pass the answer.

        :return: float
        '''
        # logger.info('Starting SEMscore calculation')
        em = self.em
        try:
            if isinstance(reference, str) and isinstance(candidate, str):
                cosine_similarity = em.get_similarities(
                    em.get_embeddings([reference]),
                    em.get_embeddings([candidate])
                )
                return round(cosine_similarity[0], 2)
            elif isinstance(reference, list):
                sum_cs = 0
                if isinstance(candidate, str):
                    candidate = [candidate]
                for doc in reference:
                    cosine_similarity = em.get_similarities(
                        em.get_embeddings([doc]),
                        em.get_embeddings(candidate)
                    )
                    sum_cs += cosine_similarity[0]
                avg_cs = sum_cs / len(reference)
                return round(avg_cs, 2)
        except Exception as e:
            # logger.info('SEMscore calculation failed\nError:', e)
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        del self.em
        self.em = None
        gc.collect()
        torch.cuda.empty_cache()