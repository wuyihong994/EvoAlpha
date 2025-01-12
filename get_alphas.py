import json
import openai
import tiktoken
from bs4 import BeautifulSoup


openai.api_key = "your_openai_api_key"


def read_template(template_file_path):
    with open(template_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'lxml')
    return html_content, soup


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


def get_response_extract(response):
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


def build_prompt_for_crossover_and_mutate_xml(alpha1_description, alpha2_description, model='gpt-4-1106-preview'):
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


def get_response_extract_for_no_crossover(response):
    soup = BeautifulSoup(response, 'lxml')
    analysis = soup.find('string', {'name': 'analysis'}).get_text(strip=True)
    mutate_alpha = soup.find('string', {'name': 'mutate_alpha'}).get_text(strip=True)
    mutate_explanation = soup.find('string', {'name': 'mutate_explanation'}).get_text(strip=True)
    res_dic = {
        "analysis": analysis,
        "mutate_alpha": mutate_alpha,
        "mutate_explanation": mutate_explanation
    }
    return analysis, mutate_alpha, mutate_explanation, res_dic


def build_prompt_for_no_crossover_xml(alpha1_description, model='gpt-4-1106-preview'):
    html_content, soup = read_template('prompts/prompt_no_crossover.html')
    system_message = soup.find_all('div', class_='message', role='system')[0].get_text(strip=False)
    pre_definition = soup.find_all('div', class_='pre-definition')[0].get_text(strip=False)
    task_definition = soup.find_all('div', class_='task definition')[0].get_text(strip=False)
    genetic_algorithm = soup.find_all('div', class_='genetic algorithm')[0].get_text(strip=False)
    functions_operators = soup.find_all('div', class_='functions and operators')[0].get_text(strip=False)
    input_data_description = soup.find_all('div', class_='input data description')[0].get_text(strip=False)
    prompt = (soup.find_all('div', class_='prompt')[0].get_text(strip=False)
              .replace("{alpha1_description}", alpha1_description))
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


def get_response_extract_for_no_mutation(response):
    soup = BeautifulSoup(response, 'lxml')
    analysis = soup.find('string', {'name': 'analysis'}).get_text(strip=True)
    crossover_alpha = soup.find('string', {'name': 'crossover_alpha'}).get_text(strip=True)
    crossover_explanation = soup.find('string', {'name': 'crossover_explanation'}).get_text(strip=True)
    res_dic = {
        "analysis": analysis,
        "crossover_alpha": crossover_alpha,
        "crossover_explanation": crossover_explanation,
    }
    return analysis, crossover_alpha, crossover_explanation, res_dic


def build_prompt_for_no_mutation_xml(alpha1_description, alpha2_description, model='gpt-4-1106-preview'):
    html_content, soup = read_template('prompts/prompt_no_mutation.html')
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


def get_response_extract_for_no_analysis(response):
    soup = BeautifulSoup(response, 'lxml')
    crossover_alpha = soup.find('string', {'name': 'crossover_alpha'}).get_text(strip=True)
    crossover_explanation = soup.find('string', {'name': 'crossover_explanation'}).get_text(strip=True)
    mutate_alpha = soup.find('string', {'name': 'mutate_alpha'}).get_text(strip=True)
    mutate_explanation = soup.find('string', {'name': 'mutate_explanation'}).get_text(strip=True)
    res_dic = {
        "crossover_alpha": crossover_alpha,
        "crossover_explanation": crossover_explanation,
        "mutate_alpha": mutate_alpha,
        "mutate_explanation": mutate_explanation
    }
    return crossover_alpha, crossover_explanation, mutate_alpha, mutate_explanation, res_dic


def build_prompt_for_no_analysis_xml(alpha1_description, alpha2_description, model='gpt-4-1106-preview'):
    html_content, soup = read_template('prompts/prompt_no_analysis.html')
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


