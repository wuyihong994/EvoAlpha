import argparse
import os
import random
from ablation_study import *
from alphagen_qlib.calculator import QLibStockDataCalculator


def get_qlib_train_test_data(dataset, data_type='train'):
    instruments = dataset
    start_time = None
    end_time = None
    if data_type == 'train':
        start_time = '2009-01-01'
        end_time = '2018-12-31'
    elif data_type == 'test':
        start_time = '2021-01-01'
        end_time = '2023-12-31'
    data = StockData(instrument=instruments,
                     start_time=start_time,
                     end_time=end_time)
    return data, start_time, end_time


class EvolutionaryAlgorithm:
    def __init__(self, file_dir, fitness_method='Pearson_IC', selection_strategy='roulette', update_strategy='cut',
                 method='GA', ablation_study='', logger=None, data_type='train'):
        # 0. Initialization
        self.continue_alphas_dict = None
        self.alphas_indicator_dict = None
        self.start_day = None
        self.end_day = None
        self.df_train = None
        self.train_target_value = None
        self.train_stock_ids = None
        self.train_dates = None
        self.train_df = None

        self.calculator_train = None
        self.data_train = None
        self.alpha_index = None
        self.mutation_rate = None
        self.population_size = None
        self.crossover_rate = None
        self.max_generations = None
        self.alphas_fitness_dict = None
        self.alphas_dict = None
        self.method = method
        self.population = []
        self.data_type = data_type
        self.logger = logger

        # 1. Load Training Set
        self.load_train_test_data()
        self.llm_alphas_value_dict = {}
        self.llm_alphas_population_dict = {}
        self.df_for_sample = self.df_train

        # 2. Load alpha dict
        self.alpha_dict = load_alpha_dict()

        # 3. EAs Config
        self.fitness_method = fitness_method
        self.selection_strategy = selection_strategy
        self.update_strategy = update_strategy
        self.ablation_study = ablation_study
        self.file_dir = file_dir
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if self.fitness_method == 'Pearson_IC':
            self.is_Pearson = True
        else:
            self.is_Pearson = False

    def load_train_test_data(self, holding_days=20):
        close = Feature(FeatureType.CLOSE)
        target = Ref(close, -1 - holding_days) / Ref(close, -1) - 1
        self.data_train, self.start_day, self.end_day = get_qlib_train_test_data(dataset='csi300',
                                                                                 data_type=self.data_type)
        self.logger.warning(
            f'\nLoad train data from {self.start_day} to {self.end_day}\n data_type: {self.data_type}\n',
            extra={'module_name': "load_train_test_data"})
        self.calculator_train = QLibStockDataCalculator(self.data_train, target)

        self.train_df, self.train_dates, self.train_stock_ids = self.data_train.get_df_dates_stocks()
        self.train_target_value = self.calculator_train.target_value

        features = ['open', 'close', 'high', 'low', 'volume', 'vwap']
        self.df_train = {}

        for feature in features:
            self.df_train[feature] = pd.concat(
                [self.train_df.loc[(i, f'${feature}')] for i in self.train_dates], axis=1
            ).transpose()
            self.df_train[feature].columns = self.train_stock_ids
            self.df_train[feature].index = self.train_dates

    def get_df_with_target(self):
        return self.df_for_sample, self.train_target_value

    def initialize_population(self, num_population):
        if self.is_Pearson:
            self.logger.info(f'Initializing Pearson IC...',
                             extra={'module_name': "initialize_population"})
            self.alphas_dict, self.alphas_indicator_dict = get_normal_alphas(
                file_name='populations/alpha_init_performance_train_csi300_hold_20_qlib.json')
        else:
            self.logger.info(f'Initialize Rank IC...',
                             extra={'module_name': "initialize_population"})
            self.alphas_dict, self.alphas_indicator_dict = get_rank_alphas(
                file_name='populations/alpha_init_performance_train_csi300_hold_20_qlib.json')

        self.alphas_dict = dict(sorted(self.alphas_dict.items(), key=lambda item: abs(item[1]), reverse=True))

        self.alphas_fitness_dict = {}
        self.continue_alphas_dict = {}
        for key, value in self.alphas_dict.items():
            if len(self.population) >= num_population:
                break
            else:
                alpha_index = key[5:]
                self.population.append(self.alphas_indicator_dict[key]['alpha_description'])
                self.alphas_fitness_dict[self.alphas_indicator_dict[key]['alpha_description']] = value
        self.logger.info(f'self.population: {self.population}',
                         extra={'module_name': "initialize_population"})

    def crossover_and_mutate(self, parent1, parent2, generate_model_name='gpt-3.5-turbo-0125'):
        crossover_and_mutate_prompt = build_prompt_for_crossover_and_mutate_xml(parent1, parent2)
        alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution = generate_alpha_code_and_fitness(
            prompt_for_crossover_mutate=crossover_and_mutate_prompt,
            df_for_sample=self.df_for_sample,
            model=generate_model_name,
            index=self.alpha_index,
            method=self.method,
            file_dir=self.file_dir,
            target_value=self.train_target_value,
            start_day=self.start_day,
            end_day=self.end_day)
        return alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution

    def crossover_and_mutate_no_crossover(self, parent1, parent2, generate_model_name='gpt-3.5-turbo'):
        crossover_and_mutate_prompt = build_prompt_for_no_crossover_xml(parent1)
        alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution = generate_alpha_code_and_fitness(
            prompt_for_crossover_mutate=crossover_and_mutate_prompt,
            df_for_sample=self.df_for_sample,
            model=generate_model_name,
            index=self.alpha_index,
            method=self.method,
            file_dir=self.file_dir,
            prompt_method='no_crossover',
            target_value=self.train_target_value,
            start_day=self.start_day,
            end_day=self.end_day)
        return alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution

    def crossover_and_mutate_no_analysis(self, parent1, parent2, generate_model_name='gpt-3.5-turbo'):
        crossover_and_mutate_prompt = build_prompt_for_no_analysis_xml(parent1, parent2)
        alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution = generate_alpha_code_and_fitness(
            prompt_for_crossover_mutate=crossover_and_mutate_prompt,
            df_for_sample=self.df_for_sample,
            model=generate_model_name,
            index=self.alpha_index,
            method=self.method,
            file_dir=self.file_dir,
            prompt_method='no_analysis',
            target_value=self.train_target_value,
            start_day=self.start_day,
            end_day=self.end_day
        )
        return alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution

    def crossover_and_mutate_no_mutation(self, parent1, parent2, generate_model_name='gpt-3.5-turbo'):
        crossover_and_mutate_prompt = build_prompt_for_no_mutation_xml(parent1, parent2)
        alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution = generate_alpha_code_and_fitness(
            prompt_for_crossover_mutate=crossover_and_mutate_prompt,
            df_for_sample=self.df_for_sample,
            model=generate_model_name,
            index=self.alpha_index,
            method=self.method,
            file_dir=self.file_dir,
            prompt_method='no_mutation',
            target_value=self.train_target_value,
            start_day=self.start_day,
            end_day=self.end_day)
        return alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution

    def select_parent(self, population, fitness_values):
        # Select 2个parent
        selected_population = []
        if self.selection_strategy == 'random':
            # print("Random selection...")
            self.logger.info(f'Random selection...',
                             extra={'module_name': "GA"})
            selected_population = random.sample(population, 2)
        elif self.selection_strategy == 'tournament':
            # print("Tournament selection")
            self.logger.info(f'Tournament selection',
                             extra={'module_name': "GA"})
            while len(selected_population) < 2:
                tournament_candidates = random.sample(population, 2)
                winner = max(tournament_candidates,
                             key=lambda x: abs(self.alphas_fitness_dict[x]))
                if winner not in selected_population:
                    selected_population.append(winner)
        elif self.selection_strategy == 'topk':
            # print("Top-k selection")
            self.logger.info(f'Top-k selection',
                             extra={'module_name': "GA"})
            selected_population = sorted(population, key=lambda x: abs(self.alphas_fitness_dict[x]),
                                         reverse=True)[:2]
        else:
            # print("Roulette wheel selection")
            self.logger.info(f'Roulette wheel selection',
                             extra={'module_name': "GA"})
            total_fitness = sum(fitness_values)
            while len(selected_population) < 2:
                # 轮盘赌选择算法
                pick = random.uniform(0, total_fitness)
                current = 0
                for index, fitness in enumerate(fitness_values):
                    current += fitness
                    if current > pick:
                        if population[index] not in selected_population:
                            selected_population.append(population[index])
                        break
        # print(f"Select Parent Individual: {selected_population}")
        self.logger.info(f'Select Parent Individual:\n{selected_population}\n',
                         extra={'module_name': "GA"})
        parent1 = selected_population[0]
        parent2 = selected_population[1]
        return parent1, parent2

    def ablation_study_set(self, parent1, parent2, generate_model_name):
        if self.ablation_study == 'no_crossover':
            crossover_mutate_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, crossover_mutate_fitness_value, df_alphas_after_code_execution = self.crossover_and_mutate_no_crossover(
                parent1, parent2, generate_model_name)
        elif self.ablation_study == 'no_analysis':
            crossover_mutate_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, crossover_mutate_fitness_value, df_alphas_after_code_execution = self.crossover_and_mutate_no_analysis(
                parent1, parent2, generate_model_name)
        elif self.ablation_study == 'no_mutation':
            crossover_mutate_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, crossover_mutate_fitness_value, df_alphas_after_code_execution = self.crossover_and_mutate_no_mutation(
                parent1, parent2, generate_model_name)
        else:
            # crossover
            crossover_mutate_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, crossover_mutate_fitness_value, df_alphas_after_code_execution = self.crossover_and_mutate(
                parent1, parent2, generate_model_name)
        return crossover_mutate_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, crossover_mutate_fitness_value, df_alphas_after_code_execution

    def check_population_similarity(self, new_alpha, df_alphas_1):
        if new_alpha in self.llm_alphas_population_dict.keys():
            return True

        df_alphas_values_1 = df_alphas_1.values
        df_alphas_values_tensor_1 = torch.tensor(df_alphas_values_1, dtype=torch.float, device='cuda:0')
        df_alphas_values_tensor_1 = normalize_by_day(df_alphas_values_tensor_1)

        for alpha in self.population:
            if alpha not in self.llm_alphas_population_dict.keys():
                # llm_alphas_value_dict存放的是生成的新alpha
                # 如果种群中的alpha不在这个字典里面，说明是alpha101的alpha，不比较相似度
                continue
            else:
                df_alphas_2 = self.llm_alphas_population_dict[alpha]['df_alpha']
                df_alphas_values_2 = df_alphas_2.values
                df_alphas_values_tensor_2 = torch.tensor(df_alphas_values_2, dtype=torch.float, device='cuda:0')
                df_alphas_values_tensor_2 = normalize_by_day(df_alphas_values_tensor_2)

                mutual_ic = batch_pearsonr(df_alphas_values_tensor_1, df_alphas_values_tensor_2).mean().item()
                self.logger.warning(
                    f"The Mutual IC Between New Alpha {self.alpha_index}: {new_alpha} and "
                    f"Alpha {self.llm_alphas_population_dict[alpha]['alpha_index']}: {alpha} is: {mutual_ic}.",
                    extra={'module_name': "check_similarity"}
                )
                if mutual_ic >= 0.99:
                    return True
        return False

    def record_update(self, crossover_mutate_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir,
                      crossover_mutate_fitness_value, df_alphas_after_code_execution):
        if crossover_mutate_alpha is not None and code is not None and crossover_mutate_fitness_value is not None:
            if self.fitness_method == 'Pearson_IC':
                self.logger.info(
                    f"LLMs Generated a New Alpha {self.alpha_index}: {crossover_mutate_alpha}. "
                    f"The Normal IC value is: {normal_ic}.",
                    extra={'module_name': "GA"})
            else:
                self.logger.info(
                    f"LLMs Generated a New Alpha {self.alpha_index}: {crossover_mutate_alpha}. "
                    f"The Rank IC value is: {rank_ic}.",
                    extra={'module_name': "GA"})

            is_Similarity_with_popoluation = self.check_population_similarity(crossover_mutate_alpha,
                                                                              df_alphas_after_code_execution)
            if is_Similarity_with_popoluation:
                self.logger.warning(f"The Generated Alpha is too similar to the alphas in the population!",
                                    extra={'module_name': "GA"})
                return True

            self.llm_alphas_population_dict[crossover_mutate_alpha] = {
                'alpha_index': self.alpha_index, 'code': code, 'normal_ic': normal_ic, 'normal_ir': normal_ir,
                'rank_ic': rank_ic, 'rank_ir': rank_ir, 'df_alpha': df_alphas_after_code_execution}

            self.alphas_indicator_dict['alpha' + str(self.alpha_index)] = {
                'alpha_description': crossover_mutate_alpha, 'normal_ic': normal_ic, 'normal_ir': normal_ir,
                'rank_ic': rank_ic,
                'rank_ir': rank_ir}

            self.llm_alphas_value_dict['alpha' + str(self.alpha_index)] = {
                'alpha_description': crossover_mutate_alpha, 'code': code, 'normal_ic': normal_ic,
                'normal_ir': normal_ir, 'rank_ic': rank_ic,
                'rank_ir': rank_ir}

            self.population.append(crossover_mutate_alpha)
            # self.alphas_fitness_dict[crossover_mutate_alpha] = crossover_mutate_fitness_value
            if self.fitness_method == 'Pearson_IC':
                self.alphas_fitness_dict[crossover_mutate_alpha] = normal_ic
            else:
                self.logger.info(
                    f"Rank ic is: {rank_ic}",
                    extra={'module_name': "GA"})
                self.alphas_fitness_dict[crossover_mutate_alpha] = rank_ic
            self.alpha_index += 1
        else:
            self.logger.error(
                f"\nError in Generating a New Alpha: {crossover_mutate_alpha}. This is may caused by alpha "
                f"generation, code generation, code execution.",
                extra={'module_name': "GA"})
        self.logger.info('------------------------', extra={'module_name': "GA"})

        return False

    def update_population(self):
        if self.update_strategy == 'replace':
            self.population = sorted(self.population, key=lambda x: abs(self.alphas_fitness_dict[x]),
                                     reverse=True)[:self.population_size]
            self.logger.info(
                f"Population is:\n{self.population}",
                extra={'module_name': "GA"})
            self.logger.info('------------------------', extra={'module_name': "GA"})

        if self.update_strategy == 'cut':
            self.population = sorted(self.population, key=lambda x: abs(self.alphas_fitness_dict[x]), reverse=True)[
                              :self.population_size]
        elif self.update_strategy == 'keep':
            self.population = sorted(self.population, key=lambda x: abs(self.alphas_fitness_dict[x]), reverse=True)
        self.logger.info(f"Population is:\n{self.population}", extra={'module_name': "GA"})
        self.logger.info('------------------------', extra={'module_name': "GA"})

    def GA(self, num_population=10, generate_model_name='gpt-3.5-turbo-0125', max_generations=10):

        # 1. Initialization
        self.initialize_population(num_population)
        population = copy.deepcopy(self.population)

        # 2. Define Config of GA
        self.max_generations = max_generations  # Max Generations
        self.population_size = len(self.population)  # Population Size

        self.logger.info(f'Initialize successfully.',
                         extra={'module_name': "GA"})
        self.alpha_index = 102
        self.logger.info(f'Alpha index is: {self.alpha_index}',
                         extra={'module_name': "GA"})

        # 3. Start Evolution
        population_list = [population]
        for generation in tqdm(range(self.max_generations), desc='Generation'):
            tmp_alpha_index = self.alpha_index
            offspring_population = []
            # 3.0 Generate 10 alphas each generation
            while (self.alpha_index - tmp_alpha_index) < 10:
                self.logger.info(f'Length of population: {self.population_size}',
                                 extra={'module_name': "GA"})
                # 3.1 fitness
                fitness_values = [abs(self.alphas_fitness_dict[individual]) for individual in population]
                self.logger.info(f'Fitness values:\n{fitness_values}',
                                 extra={'module_name': "GA"})

                # 3.2 Select parent alphas
                parent1, parent2 = self.select_parent(population, fitness_values)

                # 3.3 Start to generate alphas
                crossover_mutate_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, crossover_mutate_fitness_value, df_alphas_after_code_execution = (
                    self.ablation_study_set(parent1, parent2, generate_model_name))

                # 3.4 Record
                self.record_update(crossover_mutate_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir,
                                   crossover_mutate_fitness_value, df_alphas_after_code_execution)

                if self.update_strategy == 'replace':
                    self.update_population()

            # 3.5 Update Population
            if self.update_strategy == 'cut' or self.update_strategy == 'keep':
                self.update_population()

            population = copy.deepcopy(self.population)
            population_list.append(population)
            fw = jsonlines.open(
                f'{self.file_dir}/GA_LLM_{generation}_{self.population_size}_{generate_model_name}.json',
                'w')
            fw.write(self.llm_alphas_value_dict)
            fw.close()

        best_individual = population[0]

        self.logger.info(f'Best individual: {best_individual} \n Population is: {population}',
                         extra={'module_name': "GA"})

        fw = jsonlines.open(
            f'{self.file_dir}/GA_LLM_{self.max_generations}_{self.population_size}_{generate_model_name}_population.json',
            'w')
        fw.write(population_list)
        fw.close()


def train_and_evaluate(method, ablation_study, file_dir, fitness_method, selection_strategy, update_strategy,
                       data_type, generate_model_name, max_generations, population_size):
    logger = Logger().get_logger()
    logger.info(f'"Start..."', extra={'module_name': "main"})
    ea = EvolutionaryAlgorithm(method=method,
                               ablation_study=ablation_study,
                               file_dir=file_dir,
                               fitness_method=fitness_method,
                               selection_strategy=selection_strategy,
                               update_strategy=update_strategy,
                               logger=logger,
                               data_type=data_type)
    ea.GA(generate_model_name=generate_model_name, max_generations=max_generations, num_population=population_size)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the evolutionary algorithm.")
    parser.add_argument('--method', type=str, default='GA', help='Method for the evolutionary algorithm.')
    parser.add_argument('--ablation_study', type=str, default='all', help='Type of ablation study.')
    parser.add_argument('--file_dir', type=str, default='data_llms/15years/tmp',
                        help='Directory to store files.')
    parser.add_argument('--fitness_method', type=str, default='Pearson_IC', help='Fitness method to be used.')
    parser.add_argument('--selection_strategy', type=str, default='roulette', help='Selection strategy to be used.')
    parser.add_argument('--update_strategy', type=str, default='cut', help='Update strategy to be used.')
    parser.add_argument('--data_type', type=str, default='train', help='Type of data.')
    parser.add_argument('--generate_model_name', type=str, default='gpt-3.5-turbo-0125',
                        help='Generate Model Name to be used.')
    parser.add_argument('--max_generations', type=int, default=10, help='The Iterations of EA')
    parser.add_argument('--population_size', type=int, default=10, help='The Size of Population')

    args = parser.parse_args()

    if not os.path.exists(args.file_dir):
        os.makedirs(args.file_dir)
    print(args)

    train_and_evaluate(args.method, args.ablation_study, args.file_dir, args.fitness_method, args.selection_strategy,
                       args.update_strategy, args.data_type, args.generate_model_name, args.max_generations, args.population_size)


if __name__ == "__main__":
    main()
