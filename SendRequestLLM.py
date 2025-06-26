import gc
import time
import warnings
import pandas as pd
import GPUtil
import torch
from LLMmodel import model

warnings.filterwarnings('ignore')
class GenerateResponse:
    def __init__(self, model):
        self.model = model

    def generate(self, code, sampling_temperature=0):
        # generate response
        try:
            outputs = self.model.generate(
                text=code,
                max_length=20,
                include_prompt_in_result=False,
                sampling_temperature=sampling_temperature
            )
        except RuntimeError as e:
            return None
        gc.collect()
        torch.cuda.empty_cache()
        return outputs


generate_response = GenerateResponse(model)
