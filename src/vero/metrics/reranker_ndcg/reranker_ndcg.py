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
        # logger.info('Exception occured during rank calculation\nError:', e)
        return None


class RerankerNDCG(MetricBase):
    '''
        Implementation of NDCG@k for reranker performance measure.

        :param reranked_docs: Pass the list of reranked documents list.
        :param ground_truth: Pass the ground truth list for comparison.
        :param k: Pass the value of k for NDCG.

        Methods
        ---------
        1. __init__(reranked_docs, ground_truth, k)
            Initializes the metric.
        2. evaluate() -> list
            Returns the ndcg@k score.

        :returns: list
        '''
    name = 'reranker_ndcg'

    def __init__(self,reranked_docs: list, ground_truth: list, k:int=0):
        self.reranked_docs = reranked_docs
        self.ground_truth = ground_truth
        self.k = k

    # ndcg@k - standard for evaluating reranking
    def evaluate(self) -> list | None:
        # logger.info(f'Starting ndcg@k calculation...')
        ranks = true_ranks(self.reranked_docs, self.ground_truth)
        k = self.k
        ndcg_k = []
        try:
            for rank in ranks:
                dcg = 0
                rank_s = sorted(rank, reverse=True)
                if k != 0 and k < len(rank):
                    rank = rank[:k]
                for i in range(len(rank)):
                    dcg += (2 ** (rank[i]) - 1) / math.log2(2 + i)
                idcg = 0
                if k != 0:
                    limit = min(k, len(rank_s))
                else:
                    limit = len(rank_s)
                for i in range(limit):
                    idcg += (2 ** (rank_s[i]) - 1) / math.log2(2 + i)
                if idcg == 0:
                    ndcg_k.append(0)
                else:
                    ndcg_k.append(round((dcg / idcg), 2))

            return ndcg_k
        except Exception as e:
            # logger.info('Exception occured during NDCG@k calculation\nError:', e)
            print('Exception occured during Reranker NDCG calculation\nError:', traceback.format_exc())
            return None
