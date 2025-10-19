import traceback

from vero.metrics import MetricBase
import gc
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, BertModel
from .bleurt_pytorch import BleurtConfig,BleurtTokenizer, BleurtForSequenceClassification
import torch
# from tracing_components import logger


# Check if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

class BleurtScore(MetricBase):
    '''
        A unique implementation of BluertScore.

        Methods
        ---------
        1.  __init__()
            Initializes the models for metric.

        2. evaluate(reference,candidate) -> float
            Returns the bleurt score.

        :returns: float
    '''
    name = 'bleurt_score'

    def __init__(self):
        #TODO: change the hosted model
        config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
        self.bleurt_model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20')
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bleurt_model.to(device)
        self.bert_model.to(device)

    def __enter__(self):
        return self

    # bleurtscore - https://g.co/gemini/share/058b7af18115

    def evaluate(self,reference: str | list, candidate: str | list) -> float | None:
        '''
        :param reference: Pass the reference chunks.
        :param candidate: Pass the answer.

        :return: float
        '''
        bleurt_model = self.bleurt_model
        bleurt_tokenizer = self.bleurt_tokenizer
        bert_tokenizer = self.bert_tokenizer
        bert_model = self.bert_model

        def chunk_text(text, max_tokens=200):
            words = text.split()
            return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

        if isinstance(reference, str):
            long_reference = [reference]
        else:
            long_reference = reference
        max_tokens = 300
        reference_chunks = []

        try:
            # logger.info('Starting Bleurtscore calculation')
            for ref in long_reference:
                if len(ref.split()) > max_tokens:
                    reference_chunks += chunk_text(ref, max_tokens)
                else:
                    reference_chunks += [ref]

            if isinstance(candidate, str):
                all_text = reference_chunks + [candidate]
            else:
                all_text = reference_chunks + candidate
            bert_encodings = bert_tokenizer.batch_encode_plus(
                all_text,
                padding=True,
                truncation=True,
                return_tensors='pt')

            input_ids = bert_encodings['input_ids'].to(device)
            attention_mask = bert_encodings['attention_mask'].to(device)

            with torch.no_grad():
                output = bert_model(input_ids, attention_mask)
                word_embeddings = output.last_hidden_state

            sentence_emb = torch.mean(word_embeddings, dim=1)
            generated_emb = sentence_emb[-1]
            reference_emb = sentence_emb[:-1]
            weights_list = []

            for ref in reference_emb:
                weights_list.append(cosine_similarity(ref.unsqueeze(0), generated_emb.unsqueeze(0)).item())

            weights = [max(0, w) for w in weights_list]

            bleurt_model.eval()
            scores = []
            with torch.no_grad():
                for ref in reference_chunks:
                    inputs = bleurt_tokenizer([ref], [candidate], padding=True, truncation=True, max_length=512,
                                              return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    res = bleurt_model(**inputs).logits.flatten().tolist()
                    scores.append(res[0])

            weighted_sum = sum(s * w for w, s in zip(weights, scores))
            total_weight = sum(weights)
            final_score = weighted_sum / total_weight
            return round(final_score, 2)

        except Exception as e:
            print('BLEURT score calculation failed\nError:', traceback.format_exc())
            # logger.error('BLEURT score calculation failed\nError:', e)
            return None


    def __exit__(self, exc_type, exc_value, traceback):
        del self.bert_model
        del self.bleurt_model
        del self.bleurt_tokenizer
        del self.bert_tokenizer
        self.bert_model = None
        self.bleurt_model = None
        self.bleurt_tokenizer = None
        self.bert_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()