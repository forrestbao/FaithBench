from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

from datasets import load_dataset
from typing import Any, Generator, List
import torch
import argparse
import numpy as np
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# MODEL_NAME = 'google/t5_11b_trueteacher_and_anli'
# tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
# model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
MODEL_NAME = 'google/t5_xxl_true_nli_mixture'
tokenizer = AutoTokenizer.from_pretrained("google/t5_xxl_true_nli_mixture")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5_xxl_true_nli_mixture").to(device)


def _create_batch(source: List[str], summary: List[str], batch_size: int=1) -> Generator:
    l = len(source)
    for ndx in range(0, l, batch_size):
        batch = []
        for i in range(ndx, min(ndx + batch_size, l)):
            batch.append([source[i], summary[i]])
        
        yield batch

def TrueTeacherEval(filename, model_name, batch_size=2, update=True):
    df = pd.read_csv(filename, encoding='utf-8').fillna('')
    if (not update) and (model_name in df): # store HHEM scores
        return
    scores = []
    prompt = "premise: {source} hypothesis: {summary}"
    for batch in _create_batch(df['source'].tolist(), df['summary'].tolist(), batch_size=batch_size):
        inputs = tokenizer([prompt.format(source=pair[0], summary=pair[1]) for pair in batch], 
            return_tensors='pt',
            truncation=True,
            padding="longest",
            max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_new_tokens=5)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(result)
        result = [int(score.split()[0]) if (len(score.strip()) > 0 and score.split()[0] in ['1','0']) else None for score in result]
        scores += result
    if model_name in df:
        df[model_name] = scores
    else:
        df.insert(len(df.columns), model_name, scores)
    df.to_csv(filename, mode='w', index=False, header=True)
    print(f'{model_name} Scores have been saved')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process HHEM URL, Output and Input File paths.')
    # parser.add_argument('--filename', type=str, default="")
    # args = parser.parse_args()
    complete_df = pd.read_csv('../leaderboard_results/leaderboard_summaries.csv', encoding='utf-8')
    models = set(complete_df['model'].values.tolist())
    print(models)
    print(len(models))
    for idx, model_name in enumerate(models):
        filename = model_name + '.csv'
        # filename = filename.replace('Anthropic', 'anthropic')
        print(f"Processing file {str(idx)}: {filename} ......")
        TrueTeacherEval(filename, MODEL_NAME, batch_size=2, update=False)
        print(f"Finshed {filename}")
        print('='*20)
    
