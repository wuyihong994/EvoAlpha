import copy
import pandas as pd
import numpy as np
import tqdm
import jsonlines
import os
import warnings

from alphagen.data.expression import *
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr

from ablation_study import execute_and_evaluate_code_block_for_alpha

warnings.filterwarnings("ignore")


def get_qlib_stock_data(dataset, holding_days=20, start_time='2009-01-01', end_time='2023-12-31'):
    instruments = dataset
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -1 - holding_days) / Ref(close, -1) - 1
    data_train = StockData(instrument=instruments,
                           start_time=start_time,
                           end_time=end_time,
                           max_future_days=40)
    calculator_train = QLibStockDataCalculator(data_train, target)

    df_train, dates, stock_ids = data_train.get_df_dates_stocks()
    return df_train, dates, stock_ids, calculator_train.target_value


def load_alphas_indicators(file_name):
    fr = jsonlines.open(file_name, "r")
    alphas_indicator_dict = {}
    for line in fr:
        for key, value in line.items():
            alphas_indicator_dict[key] = value
    fr.close()
    return alphas_indicator_dict


def load_alpha_dict():
    fr_path = f"populations/alphas101.json"
    fr = jsonlines.open(fr_path, "r")
    alpha_dict = {}
    for line in fr:
        for key, value in line.items():
            alpha_dict[key] = value
    fr.close()
    return alpha_dict


def evaluate_ours_qlib(path, dataset, holding_days, generate_model_name='gpt-3.5-turbo-0125', iteration=9, population_size=10, is_Train=False, is_Valid=False):
    if is_Train:
        start_date = '2009-01-01'
        end_date = '2018-12-31'
    elif is_Valid:
        start_date = '2019-01-01'
        end_date = '2020-12-31'
    else:
        start_date = '2021-01-01'
        end_date = '2023-12-31'

    df_data, df_index, stock_list, target_value = get_qlib_stock_data(dataset, holding_days, start_time=start_date,
                                                                      end_time=end_date)

    features = ['open', 'close', 'high', 'low', 'volume', 'vwap']
    df_test = {}

    for feature in features:
        df_test[feature] = pd.concat(
            [df_data.loc[(i, f'${feature}')] for i in df_index], axis=1
        ).transpose()
        df_test[feature].columns = stock_list
        df_test[feature].index = df_index

    fr_path = f"{path}/GA_LLM_{iteration}_{population_size}_{generate_model_name}.json"
    fr = jsonlines.open(fr_path, "r")
    alphas_dict = {}
    for line in fr:
        for key, value in line.items():
            alphas_dict[key] = value
    fr.close()

    alphas_fitness = copy.deepcopy(alphas_dict)
    for alpha in alphas_dict.keys():
        code = alphas_dict[alpha]['code']

        e, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, _ = execute_and_evaluate_code_block_for_alpha(
            "", code, df_test, target_value, start_date, end_date
        )
        alphas_fitness[alpha]['normal_ic'] = normal_ic
        alphas_fitness[alpha]['normal_ir'] = normal_ir
        alphas_fitness[alpha]['rank_ic'] = rank_ic
        alphas_fitness[alpha]['rank_ir'] = rank_ir

    if is_Train:
        file_name = f'{path}/alpha_performance_train_{dataset}_hold_{holding_days}_iteration_{iteration}_qlib.json'
    elif is_Valid:
        file_name = f'{path}/alpha_performance_valid_{dataset}_hold_{holding_days}_iteration_{iteration}_qlib.json'
    else:
        file_name = f'{path}/alpha_performance_test_{dataset}_hold_{holding_days}_iteration_{iteration}_qlib.json'

    fw = jsonlines.open(file_name, "w")
    fw.write(alphas_fitness)
    fw.close()
