from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import pandas as pd
from alphagen.data.expression import *
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from logger import *

logger = Logger().get_logger()

def calculate_fitness(df_alphas, target_value, start_day, end_day):
    try:
        df_alphas = df_alphas[(df_alphas.index >= start_day) & (df_alphas.index <= end_day)]
        df_alphas_values = df_alphas.values
        df_alphas_values_tensor = torch.tensor(df_alphas_values, dtype=torch.float, device='cuda:0')

        df_alphas_values_tensor = normalize_by_day(df_alphas_values_tensor)

        normal_ic = batch_pearsonr(df_alphas_values_tensor, target_value).mean().item()
        normal_ic_std = batch_pearsonr(df_alphas_values_tensor, target_value).std().item()
        normal_ir = normal_ic / normal_ic_std if normal_ic_std != 0 else 0

        torch.cuda.empty_cache()

        rank_ic = batch_spearmanr(df_alphas_values_tensor, target_value).mean().item()
        rank_ic_std = batch_spearmanr(df_alphas_values_tensor, target_value).std().item()
        rank_ir = rank_ic / rank_ic_std if rank_ic_std != 0 else 0

        fitness_value = normal_ic

        logger.debug(
            f"Normal IC:  {normal_ic}, Rank IC:  {rank_ic}",
            extra={'module_name': "calculate_fitness"})

        return normal_ic, normal_ir, rank_ic, rank_ir, fitness_value
    except Exception as e:
        return None, None, None, None, None
