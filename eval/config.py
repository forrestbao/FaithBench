import os

current_folder = os.path.dirname(os.path.abspath(__file__))
REF_FILES = [f'{os.path.dirname(current_folder)}/assign/examples_to_annotate.csv']
RESULT_PATH = f'{os.path.dirname(current_folder)}/assign/batch_5_src_no_sports/results'
DETECTORS = {
    "HHEMv1":"HHEM-1", 
    "HHEM-2.1": "HHEM-2.1-Tri" , 
    "HHEM-2.1-English": "HHEM-2.1-English", 
    "HHEM-2.1-Open": "HHEM-2.1-Open",
    "alignscore-base": "AlignScore-BS",
    "alignscore-large": "AlignScore-LG",
    "trueteacher": "True-Teacher", 
    "true_nli": "True-NLI", 
    # "gpt-3.5-turbo": "GPT-3.5-Turbo, zero-shot",
    # "gpt-3.5-turbo-FACTSprompt": "GPT-3.5-Turbo, FACTS", 
    "gpt-4-turbo": "GPT-4-Turbo, zero-shot", 
    # "gpt-4-turbo-FACTSprompt": "GPT-4-Turbo, FACTS",
    "gpt-4o": "GPT-4o, zero-shot",
    # "gpt-4o-FACTSprompt": "GPT-4o, FACTS", 
    # "gpt-4": "GPT-4, zero-shot",
    # "gpt-4-FACTSprompt": "GPT-4, FACTS",
    "o1-mini": "O1-Mini, zero-shot",
    # "o1-mini-FACTSprompt": "O1-Mini, FACTS",
    "o3-mini": "O3-Mini, zero-shot",
    # "o3-mini-FACTSprompt": "O3-Mini, FACTS",
    "minicheck-roberta-large": "Minicheck-Roberta-LG",
    "minicheck-deberta-v3-large": "Minicheck-Deberta-LG",
    "minicheck-flan-t5-large": "Minicheck-Flan-T5-LG",
    "Ragas_gpt-4o": "Ragas-GPT-4o",
    "Trulens_gpt-4o_scores": "Trulens-GPT-4o",
}

SUMMARY_SENT_FILE = f'{current_folder}/sent_level_results/summary_sent_list.jsonl'