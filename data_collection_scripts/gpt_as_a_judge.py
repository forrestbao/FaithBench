import re
import os
import time
import pandas as pd
from openai import OpenAI

client = OpenAI()
MODEL='gpt-4o'
label_mapping = {'yes':1, 'no':0}

# Aggrefact prompt
system = 'Decide if the following summary is consistent with the correponding article. Note that consistency means all information in the summary is supported by the article.'
user = 'Article: {article}\nSummary: {summary}\nAnswer (yes or no):'

def call_gpt(system_prompt, user_prompt, model='gpt-4', temperature=0):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )

    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    print(current_folder)
    subfolders = [f.path for f in os.scandir(current_folder) if f.is_dir()]
    model_files = []
    for subfolder in subfolders:
        model_files += [f.path for f in os.scandir(subfolder) if f.is_file()]
    print(model_files)
    selected_models = [
        "openai/GPT-3.5-Turbo",
        "openai/gpt-4o",
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "cohere/command-r-08-2024",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "google/gemini-1.5-flash-001",
        "Anthropic/claude-3-5-sonnet-20240620",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    for file_name in model_files:
        print(file_name.replace(current_folder,'')[1:].replace('.csv',''))
        if file_name.replace(current_folder,'')[1:].replace('.csv','') not in selected_models:
            continue
        with open(file_name) as f:
            print(file_name)
            df = pd.read_csv(file_name).fillna('')
            preds = []
            for index, row in df.iterrows():
                # start_time = time.time()
                result = call_gpt(system, user.format(article=row['source'], summary=row['summary']), model=MODEL)
                result = re.sub(r'[^\w\s]', '', result)
                print(label_mapping[result.strip().lower().split()[0]])
                preds.append(label_mapping[result.strip().lower().split()[0]])
                
            df.insert(len(df.columns.tolist()), MODEL, preds)
            df.to_csv(file_name, mode='w', index=False, header=True)
    
