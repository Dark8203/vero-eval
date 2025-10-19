import traceback

from vero.metrics import MetricBase
import numpy as np
import math
# from rag_code.tracing_components import logger


class MeanRR(MetricBase):
    '''
        Implementation of MRR for reranker performance measure.

        :param reranked_docs: Pass the list of reranked documents list.
        :param ground_truth: Pass the ground truth list for comparison.

        Methods
        ---------
        1. __init__(reranked_docs, ground_truth, k)
            Initializes the metric.
        2. evaluate() -> list
            Returns the mrr score.

        :returns: float
        '''
    name = 'mean_reciprocal_rank'

    def __init__(self,reranked_docs: list, ground_truth: list):
        self.reranked_docs = reranked_docs
        self.ground_truth = ground_truth

    def evaluate(self) -> float | None:
        # logger.info('Starting MRR calculation...')
        reciprocal_rank = []
        count = 0
        try:
            for docs, truth in zip(self.reranked_docs, self.ground_truth):
                count += 1
                for i in range(len(docs)):
                    if docs[i] in truth:
                        reciprocal_rank.append(1 / (i + 1))
                        break
                if len(reciprocal_rank) < count:
                    reciprocal_rank.append(0)

            mrr = round((sum(reciprocal_rank) / len(reciprocal_rank)), 2)
            return mrr
        except Exception as e:
            # logger.info('Exception occured during MRR calculation\nError:', e)
            print('Exception occured during MRR calculation\nError:', traceback.format_exc())
            return None