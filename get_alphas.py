import json
import openai
import tiktoken
from bs4 import BeautifulSoup

from logger import *

logger = Logger().get_logger()

openai.api_key = "your_openai_api_key"

# =============================================================================
# Some utility functions
def read_template(template_file_path):
    with open(template_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'lxml')
    return html_content, soup

# ============================================================================
# Evolutionary CoT for New Alpha Generation

# build prompt for LLM-based evolution
def build_prompt_for_evolution(alpha1_description, alpha2_description):
    html_content, soup = read_template('prompts/prompt.html')
    system_message = soup.find_all('div', class_='message', role='system')[0].get_text(strip=False)
    pre_definition = soup.find_all('div', class_='pre-definition')[0].get_text(strip=False)
    task_definition = soup.find_all('div', class_='task definition')[0].get_text(strip=False)
    genetic_algorithm = soup.find_all('div', class_='genetic algorithm')[0].get_text(strip=False)
    functions_operators = soup.find_all('div', class_='functions and operators')[0].get_text(strip=False)
    input_data_description = soup.find_all('div', class_='input data description')[0].get_text(strip=False)
    prompt = (soup.find_all('div', class_='prompt')[0].get_text(strip=False)
              .replace("{alpha1_description}", alpha1_description)
              .replace("{alpha2_description}", alpha2_description))
    output_format = soup.find_all('div', class_='output_format')[0].get_text(strip=False)

    system_prompt = [
        {"type": "text", "text": system_message}
    ]
    user_prompt = [
        {"type": "text", "text": content}
        for content in [
            pre_definition,
            task_definition,
            genetic_algorithm,
            functions_operators,
            input_data_description,
            prompt,
            output_format
        ]
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages

# get response from LLM
def get_filled_template_response(messages, model="gpt-3.5-turbo"):
    if type(messages) is not list:
        messages = [
            {'role': 'system', 'content': 'You are a professor.'},
            {'role': 'user', 'content': messages}
        ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1.0,
    )

    total_tokens = response['usage']['total_tokens']
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    model_name = response['model']
    print(f'Total_tokens_used: {prompt_tokens} + {completion_tokens} = {total_tokens} | {model_name}')

    result = ''
    for choice in response.choices:
        result += choice.message.content
    return result

# extract results from LLM response
def extract_from_response(response):
    soup = BeautifulSoup(response, 'lxml')
    analysis = soup.find('string', {'name': 'analysis'}).get_text(strip=True)
    crossover_alpha = soup.find('string', {'name': 'crossover_alpha'}).get_text(strip=True)
    crossover_explanation = soup.find('string', {'name': 'crossover_explanation'}).get_text(strip=True)
    mutate_alpha = soup.find('string', {'name': 'mutate_alpha'}).get_text(strip=True)
    mutate_explanation = soup.find('string', {'name': 'mutate_explanation'}).get_text(strip=True)
    res_dic = {
        "analysis": analysis,
        "crossover_alpha": crossover_alpha,
        "crossover_explanation": crossover_explanation,
        "mutate_alpha": mutate_alpha,
        "mutate_explanation": mutate_explanation
    }
    return analysis, crossover_alpha, crossover_explanation, mutate_alpha, mutate_explanation, res_dic

# generate new alpha via Evolutionary CoT, which leverages LLMs to perform Evolutionary Operations
def generate_alpha(messages, model, alpha_response_list):
    try_num = 0
    new_alpha = None
    while try_num < 2:
        try_num += 1
        try:
            alpha_response = get_filled_template_response(messages, model=model)
            alpha_response_list.append(alpha_response)

            analysis, crossover_alpha, crossover_explanation, mutate_alpha, mutate_explanation, result_dict = extract_from_response(
                alpha_response)

            new_alpha = mutate_alpha

            logger.info(
                f'\n|-Analysis-|:\n{analysis}\n|-Crossover_alpha-|:\n{crossover_alpha}\n|-Crossover_explanation-|:\n{crossover_explanation}\n|-Mutate_alpha-|:\n{mutate_alpha}\n|-Mutate_explanation-|:\n{mutate_explanation}\n',
                extra={'module_name': "generate_alpha"})

            logger.warning(f'\nGenerated alpha after crossover and mutation:\n{new_alpha}\n',
                           extra={'module_name': "generate_alpha"})

        except Exception as e:
            logger.error(f"\nError in LLM Generating a New Alpha.\n{e}",
                         extra={'module_name': "generate_alpha"})
            continue

        if new_alpha is None:
            messages += [
                {
                    "role": "user",
                    "content": f"""Alpha generation failed with error. Please make sure the generated alpha is 
                    bracketed with <mutate_alpha> and </mutate_alpha>.\n Generate alpha again:""",
                },
            ]
        else:
            break
    return new_alpha, alpha_response_list

def get_alpha(alpha1_description, alpha2_description, model="gpt-3.5-turbo"):
    messages = build_prompt_for_evolution(alpha1_description, alpha2_description)
    alpha_response_list = []

    # 1. Generate new alpha
    new_alpha, alpha_response_list = generate_alpha(messages, model, alpha_response_list)
    return new_alpha, alpha_response_list

if __name__ == "__main__":
    alpha1 = "your_alpha_description_1"
    alpha2 = "your_alpha_description_2"
    model = "your_model_name_here"

    new_alpha, alpha_response_list = get_alpha(alpha1, alpha2, model)



