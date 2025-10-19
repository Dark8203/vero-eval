import traceback

from vero.metrics import MetricBase
import numpy as np
import math
# from rag_code.tracing_components import logger

def true_ranks(reranked_docs: list, ground_truth: list) -> list | None:
    # logger.info('Calculating ranks...')
    try:
        ranks = []
        for re_doc, truth in zip(reranked_docs, ground_truth):
            s = []
            for doc in re_doc:
                if doc in truth:
                    s.append(truth[doc])
                else:
                    s.append(0)
            ranks.append(s)
        return ranks
    except Exception as e:
        print('Exception occured during rank calculation\nError:', traceback.format_exc())
        # logger.info('Exception occured during rank calculation\nError:', e)
        return None


class CumulativeNDCG(MetricBase):
    '''
        Unique implementation of NDCG that can be used to evaluate the cumulative performance of retriever and reranker.

        :param reranked_docs: Pass the list of reranked documents list.
        :param ground_truth: Pass the ground truth list for comparison.

        Methods
        ---------
        1. __init__(reranked_docs, ground_truth)
            Initializes the metric.
        2. evaluate() -> list
            Returns the ndcg score.

        :returns: list
        '''
    name = 'cumulative_ndcg'

    def __init__(self,reranked_docs: list, ground_truth: list):
        self.reranked_docs = reranked_docs
        self.ground_truth = ground_truth

    def evaluate(self) -> list | None:
        # logger.info(f'Starting ndcg calculation...')
        ranks = true_ranks(self.reranked_docs, self.ground_truth)
        ndcg = []
        try:
            for rank, truth in zip(ranks, self.ground_truth):
                dcg = 0
                idcg = 0
                k = len(rank)
                for i in range(k):
                    dcg += (2 ** (rank[i]) - 1) / math.log2(2 + i)

                truth_ranks = sorted(truth.values(), reverse=True)

                limit = min(k, len(truth_ranks))
                for i in range(limit):
                    idcg += (2 ** (truth_ranks[i]) - 1) / math.log2(2 + i)
                if idcg == 0:
                    ndcg.append(0)
                else:
                    ndcg.append(round(dcg / idcg, 2))
            return ndcg
        except Exception as e:
            print('Exception occured during cumulative NDCG calculation\nError:', traceback.format_exc())
            # logger.info('Exception occured during NDCG calculation\nError:', e)
            return None