# Add sample ID to records in Miaoran's original CSV

import pandas as pd
df = pd.read_csv('examples_to_annotate.csv')
df['sample_id'] = df.index
df.to_json('examples_to_annotate.jsonl', orient='records', lines=True)