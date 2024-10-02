# Assume mercury ingest is in the path

MERCURY_PATH="../../../mercury" # the root directory of Mercury's repository
VIRGIN_PATH="./virgin_data"

python3 $MERCURY_PATH/ingester.py --embedding_model_id="bge-small-en-v1.5" --sqlite_db_path=$VIRGIN_PATH/pilot_bge_small.sqlite $VIRGIN_PATH/pilot_bge_small.jsonl
python3 $MERCURY_PATH/ingester.py --embedding_model_id="openai/text-embedding-3-small" --sqlite_db_path=$VIRGIN_PATH/pilot_openai_small.sqlite $VIRGIN_PATH/pilot_openai_small.jsonl
# python3 $MERCURY_PATH/ingester.py --embedding_model_id="all-mpnet-base-v2" --sqlite_db_path="$VIRGIN_PATH//pilot_all-mpnet-base-v2.sqlite" $VIRGIN_PATH//pilot_all-mpnet-base-v2.jsonl
