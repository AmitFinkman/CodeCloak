from abc import ABC, abstractmethod
from typing import List, Union, Dict

import torch
from transformers import AutoTokenizer, AutoModel

MASK_TOKEN = "<mask>"
SEPARATOR_TOKEN = "<sep>"
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"


def set_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    output_data = {}
    for k, v in inputs.items():
        output_data[k] = v.to(device)
    return output_data


def pooling(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eos_idx = mask.sum(1) - 1
    batch_idx = torch.arange(len(eos_idx), device=x.device)
    mu = x[batch_idx, eos_idx, :]
    return mu


def pool_and_normalize(
    features_sequence: torch.Tensor,
    attention_masks: torch.Tensor,
    return_norms: bool = False):
    pooled_embeddings = pooling(features_sequence, attention_masks)
    embedding_norms = pooled_embeddings.norm(dim=1)

    normalizing_factor = torch.where(  # Only normalize embeddings with norm > 1.0.
        embedding_norms > 1.0, embedding_norms, torch.ones_like(embedding_norms)
    )

    pooled_normalized_embeddings = pooled_embeddings / normalizing_factor[:, None]

    if return_norms:
        return pooled_normalized_embeddings, embedding_norms
    else:
        return pooled_normalized_embeddings


def truncate_sentences(sentence_list: List[str], maximum_length: Union[int, float]):
    truncated_sentences = []
    for sentence in sentence_list:
        truncated_sentences.append(sentence[:maximum_length])
    return truncated_sentences


def prepare_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.add_special_tokens({"sep_token": SEPARATOR_TOKEN})
    tokenizer.add_special_tokens({"cls_token": CLS_TOKEN})
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer


class BaseEncoder(torch.nn.Module, ABC):
    def __init__(self, device, max_input_len, maximum_token_len, model_name):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = prepare_tokenizer(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to('cuda').eval()
        self.device = device
        self.max_input_len = max_input_len
        self.maximum_token_len = maximum_token_len

    @abstractmethod
    def forward(self, text):
        pass

    def encode(self, input_sentences, batch_size=32, **kwargs):
        truncated_input_sentences = truncate_sentences(input_sentences, self.max_input_len)
        n_batches = len(truncated_input_sentences) // batch_size + int(len(truncated_input_sentences) % batch_size > 0)
        embedding_batch_list = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(truncated_input_sentences))

            with torch.no_grad():
                embedding_batch_list.append(
                    self.forward(truncated_input_sentences[start_idx:end_idx]).detach().cpu()
                )

        input_sentences_embedding = torch.cat(embedding_batch_list)

        return [emb.squeeze().numpy() for emb in input_sentences_embedding]


class StarEncoder(BaseEncoder):

    def __init__(self, device, max_input_len, maximum_token_len):
        super().__init__(device, max_input_len, maximum_token_len, model_name="bigcode/starencoder")

    def forward(self, input_sentences):
        inputs = self.tokenizer(
            [f"{CLS_TOKEN}{sentence}{SEPARATOR_TOKEN}" for sentence in input_sentences],
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.encoder(**set_device(inputs, self.device))
        embedding = pool_and_normalize(outputs.hidden_states[-1], inputs.attention_mask)
        return embedding


DEVICE = "cuda"
BATCH_SIZE = 32
MAX_INPUT_LEN = 10000
MAX_TOKEN_LEN = 1024
star_encoder = StarEncoder(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)



