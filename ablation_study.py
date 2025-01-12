import jsonlines
from get_alphas import *
from get_code import *
from utils import *
from calculator import *

err_num = 0
logger = Logger().get_logger()


def get_response_for_code(messages, model="gpt-3.5-turbo"):
    if model == "skip":
        return ""

    try_num = 0
    while try_num < 3:
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stop=["```end"],
                temperature=0.5,
                max_tokens=600,
            )
            response = completion["choices"][0]["message"]["content"]
            logger.warning(f"Get_response:\n {response}",
                           extra={'module_name': "get_response_for_code"})
            code = response
            code = code.replace("```python", "").replace("```end", "").replace("```", "").replace("<end>", "")
            return code, response
        except Exception as e:
            try_num += 1
            logger.error(f"\nAn error occurred in Openai: {str(e)}",
                         extra={'module_name': "get_response_for_code"})
    return "OpenAI API Error"


def execute_and_evaluate_code_block_for_alpha(
        full_code,
        code,
        df_for_sample,
        target_value,
        start_day,
        end_day
):
    df_tmp = df_for_sample['open']
    logger.warning(
        f'\nstart_day: {start_day}\nend_day: {end_day}\n'
        f'len(df_tmp.index) :{len(df_tmp.index)}\n',
        extra={'module_name': "execute_and_evaluate_code_block_for_alpha"}
    )
    try:

        df_before = copy.deepcopy(df_for_sample)
        df_after = run_llm_code(
            full_code + "\n" + code,
            df_before
        )
        df_alphas = df_after['alpha']

    except Exception as e:
        logger.error(
            f"\n(代码执行出错)Error in code execution.\n{type(e)} {e}\n ```python\n{format_for_display(code)}\n```\n",
            extra={'module_name': "execute_and_evaluate_code_block_for_alpha"})

        return e, None, None, None, None, None, None
    print(df_alphas)
    normal_ic, normal_ir, rank_ic, rank_ir, fitness_value = calculate_fitness(df_alphas, target_value,
                                                                              start_day, end_day)
    logger.warning(
        f'\nnormal_ic: {normal_ic}\nnormal_ir: {normal_ir}\nrank_ic: {rank_ic}\nrank_ir: {rank_ir}\nfitness_value: {fitness_value}\n',
        extra={'module_name': "execute_and_evaluate_code_block_for_alpha"}
    )
    return None, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas


def generate_alpha(messages, model, alpha_response_list, prompt_method):
    try_num = 0
    new_alpha = None
    while try_num < 2:
        try_num += 1
        try:
            alpha_response = get_filled_template_response(messages, model=model)
            alpha_response_list.append(alpha_response)
            if prompt_method == 'no_crossover':
                logger.warning(f'\n"No crossover...."\n',
                               extra={'module_name': "generate_alpha_code_and_fitness"})
                analysis, mutate_alpha, mutate_explanation, result_dict = get_response_extract_for_no_crossover(
                    alpha_response)
                new_alpha = mutate_alpha
                logger.info(
                    f'\n|-analysis-|:\n{analysis}\n|-mutate_alpha-|:\n{mutate_alpha}\n|-mutate_explanation-|:\n{mutate_explanation}\n',
                    extra={'module_name': "generate_alpha_code_and_fitness"})

            elif prompt_method == 'no_analysis':
                logger.warning(f'\n"No analysis...."\n',
                               extra={'module_name': "generate_alpha_code_and_fitness"})
                crossover_alpha, crossover_explanation, mutate_alpha, mutate_explanation, result_dict = get_response_extract_for_no_analysis(
                    alpha_response)
                new_alpha = mutate_alpha
                logger.info(
                    f'\n|-crossover_alpha-|:\n{crossover_alpha}\n|-crossover_explanation-|:\n{crossover_explanation}\n|-mutate_alpha-|:\n{mutate_alpha}\n|-mutate_explanation-|:\n{mutate_explanation}\n',
                    extra={'module_name': "generate_alpha_code_and_fitness"})

            elif prompt_method == 'no_mutation':
                logger.warning(f'\n"No mutation...."\n',
                               extra={'module_name': "generate_alpha_code_and_fitness"})
                analysis, crossover_alpha, crossover_explanation, result_dict = get_response_extract_for_no_mutation(
                    alpha_response)
                new_alpha = crossover_alpha

                logger.info(
                    f'\n|-Analysis-|:\n{analysis}\n|-Crossover_alpha-|:\n{crossover_alpha}\n|-Crossover_explanation-|:\n{crossover_explanation}\n',
                    extra={'module_name': "generate_alpha_code_and_fitness"})
            else:
                analysis, crossover_alpha, crossover_explanation, mutate_alpha, mutate_explanation, result_dict = get_response_extract(
                    alpha_response)
                if mutate_alpha != "The new alpha generated by mutation":
                    new_alpha = mutate_alpha
                logger.info(
                    f'\n|-Analysis-|:\n{analysis}\n|-Crossover_alpha-|:\n{crossover_alpha}\n|-Crossover_explanation-|:\n{crossover_explanation}\n|-Mutate_alpha-|:\n{mutate_alpha}\n|-Mutate_explanation-|:\n{mutate_explanation}\n',
                    extra={'module_name': "generate_alpha_code_and_fitness"})

            logger.warning(f'\nGenerated alpha after crossover and mutation:\n{new_alpha}\n',
                           extra={'module_name': "generate_alpha_code_and_fitness"})

        except Exception as e:
            logger.error(f"\nError in LLM Generating a New Alpha.\n{e}",
                         extra={'module_name': "generate_alpha_code_and_fitness"})
            continue

        if new_alpha is None:
            messages += [
                {
                    "role": "user",
                    "content": f"""Alpha generation failed with error. Please make sure the generated alpha is 
                    bracketed with <alpha> and </alpha>.\n Generate alpha again:""",
                },
            ]
        else:
            break
    return new_alpha, alpha_response_list


def generate_code(new_alpha, model, code_response_list, full_code, df_for_sample, target_value,
                  start_day, end_day):
    e, code = None, None
    normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution = None, None, None, None, None, None
    prompt_for_code_generation = build_prompt_from_df_for_code_generation(df_for_sample, new_alpha)
    messages_for_code_generation = [
        {
            "role": "system",
            "content": "You are an expert in quantitative finance trading, proficient and knowledgeable in "
                       "quantitative alpha factors. You task is to generate code base on the alpha. You answer only by generating code. Please answer as "
                       "concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt_for_code_generation,
        },
    ]
    try_num = 0
    while try_num < 3:
        try_num += 1
        try:
            code, code_response = get_response_for_code(messages_for_code_generation, model=model)
            code_response_list.append(code_response)
        except Exception as es:
            logger.error(f"\nError in LLM Generating Codeblock for the new alpha. \n{es}",
                         extra={'module_name': "generate_alpha_code_and_fitness"})
            continue

        assert code is not None, "Error in Code generation."

        # 0. Check Code is executable and doesn't use future information.
        code_runnable, are_equal = check_alpha_code_valid(code)
        logger.warning(f"\nThe equal test results : {are_equal}\n",
                       extra={'module_name': "generate_code"})
        # 1. Code is Unexexcutable
        if code_runnable is False:
            messages_for_code_generation += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n.Generate new code (fixing error?)(You answer only by generating code):
                                                           ```python
                                                           """,
                },
            ]
            continue

        # 2. Code is excutable But Use future information
        if are_equal is False:
            logger.error(f"\nRegenerate! The equal test results don't equal\n",
                         extra={'module_name': "generate_code"})
            messages_for_code_generation += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""The feature should only use information from the current and past data, and should not use any future data. For example, using df['open'].sum() would use all time periods' df['open'], but I want the generated code to avoid using future data points. Please ensure that the code adheres to this requirement.
                    The code you generate: \n Code: ```python{code}```\n Please generate new code, please follow my requirement to fix it:
                                                           ```python
                                                           """,
                },
            ]
            continue

        # 3. Code is excutable and doesn't use future information
        if code_runnable and are_equal:
            e, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution = execute_and_evaluate_code_block_for_alpha(
                full_code, code, df_for_sample, target_value, start_day, end_day
            )
            # 3.1 到本体数据执行产生错误
            if e is not None:
                messages_for_code_generation += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate new code (fixing error?):
                                        ```python
                                        """,
                    },
                ]
                continue
            else:
                break
    return e, code, messages_for_code_generation, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution


def generate_alpha_code_and_fitness(
        prompt_for_crossover_mutate,
        df_for_sample,
        model="gpt-3.5-turbo",
        index=0,
        method="GA",
        file_dir='',
        prompt_method='',
        target_value=None,
        start_day=None,
        end_day=None
):
    full_code = ""
    alpha_response = None
    result_dict = None
    alpha_response_list = []
    code_response_list = []

    # 1. Generate new alpha
    messages = prompt_for_crossover_mutate
    new_alpha, alpha_response_list = generate_alpha(messages, model, alpha_response_list, prompt_method)
    if new_alpha is None:
        return None, None, None, None, None, None, None, None

    # 2. Generate Corresponding Code
    e, code, messages_for_code_generation, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution = generate_code(
        new_alpha, model, code_response_list, full_code,
        df_for_sample, target_value,
        start_day, end_day)

    if e is not None or code is None:
        return new_alpha, code, None, None, None, None, None, None

    fw = open(f'{file_dir}/history/{model}_history_{method}_{index}.txt', "w")
    fw.write(f"{messages}" + '\n')
    fw.write('-' * 50 + '\n')
    fw.write(f"{alpha_response_list}\n")
    fw.write('-' * 50 + '\n')
    fw.write(f"{messages_for_code_generation}" + '\n')
    fw.write(f"{code_response_list}\n")
    fw.close()

    fw = open(f'{file_dir}/response/{model}_response_alpha_{method}_{index}.txt', "w")
    fw.write(f"{alpha_response}\n")
    fw.write('-' * 50 + '\n')
    fw.write(f"{result_dict}\n")
    fw.write('*' * 50 + '\n')
    fw.close()

    fw = open(f'{file_dir}/generated_code/{model}_alpha_{method}_{index}.txt', "w")
    fw.write(f"{new_alpha}")
    fw.close()

    fw = open(f'{file_dir}/generated_code/{model}_code_{method}_{index}.txt', "w")
    full_code_ = full_code + "\n" + code
    fw.write(f"{full_code_}")
    fw.close()

    return new_alpha, code, normal_ic, normal_ir, rank_ic, rank_ir, fitness_value, df_alphas_after_code_execution


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


def get_rank_alphas(file_name):
    alphas_indicator_dict = load_alphas_indicators(file_name=file_name)
    alphas_value_dict = {}
    for alpha in alphas_indicator_dict.keys():
        alpha_value = alphas_indicator_dict[alpha]["rank_ic"]
        if np.isnan(alpha_value):
            alpha_value = 0
        alphas_value_dict[alpha] = alpha_value
    sorted_dict = dict(sorted(alphas_value_dict.items(), key=lambda item: abs(item[1]), reverse=True))
    return sorted_dict, alphas_indicator_dict


def get_normal_alphas(file_name):
    alphas_indicator_dict = load_alphas_indicators(file_name=file_name)
    alphas_value_dict = {}
    for alpha in alphas_indicator_dict.keys():
        alpha_value = alphas_indicator_dict[alpha]["normal_ic"]
        if np.isnan(alpha_value):
            alpha_value = 0
        alphas_value_dict[alpha] = alpha_value
    sorted_dict = dict(sorted(alphas_value_dict.items(), key=lambda item: abs(item[1]), reverse=True))
    return sorted_dict, alphas_indicator_dict
