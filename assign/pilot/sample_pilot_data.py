import pandas as pd

df = pd.read_json('../examples_to_annotate.jsonl', lines=True)


N = 10 

# BGE embedding 
embed_in_bge_small = df.sample(n=N)
embed_in_bge_small.to_json('virgin_data/pilot_bge_small.jsonl', orient='records', lines=True)

# openai small embedding
embed_in_openai_small = df.sample(n=N)
embed_in_openai_small.to_json('virgin_data/pilot_openai_small.jsonl', orient='records', lines=True)

# all-mpnet-base-v2 embedding
embed_in_openai_small = df.sample(n=N)
embed_in_openai_small.to_json('vigin_data/pilot_all-mpnet_small.jsonl', orient='records', lines=True)