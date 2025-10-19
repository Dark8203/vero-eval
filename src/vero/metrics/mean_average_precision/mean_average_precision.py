import traceback

from vero.metrics import MetricBase
import numpy as np
import math
# from rag_code.tracing_components import logger


class MeanAP(MetricBase):
    '''
        Implementation of MAP for reranker performance measure.

        :param reranked_docs: Pass the list of reranked documents list.
        :param ground_truth: Pass the ground truth list for comparison.

        Methods
        ---------
        1. __init__(reranked_docs, ground_truth, k)
            Initializes the metric.
        2. evaluate() -> list
            Returns the map score.

        :returns: float
        '''
    name = 'mean_average_precision'

    def __init__(self,reranked_docs: list, ground_truth: list):
        self.reranked_docs = reranked_docs
        self.ground_truth = ground_truth

    def evaluate(self) -> float | None:
        # logger.info('Starting MAP calculation...')
        avg_precision = []
        try:
            for docs, truth in zip(self.reranked_docs, self.ground_truth):
                precesion = 0
                doc_count = 0
                truth_length = len(truth)
                for i in range(len(docs)):
                    if docs[i] in truth:
                        doc_count += 1
                        precesion += doc_count / (i + 1)
                avg_precision.append(precesion / truth_length)

            map = round(sum(avg_precision) / len(avg_precision), 2)
            return map
        except Exception as e:
            # logger.info('Exception occured during MAP calculation\nError:', e)
            print('Exception occured during MAP calculation\nError:', traceback.format_exc())
            return None