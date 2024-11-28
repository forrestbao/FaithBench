# Assume mercury ingest is in the path

MERCURY_PATH="." # the root directory of Mercury's repository
Faithbench_PATH="./FaithBench_pilot" # the root directory of Faithbench's repository

source /home/forrest/label_env/bin/activate

export SECRET_KEY=$(openssl rand -base64 32)

#python3 $MERCURY_PATH/server.py --mercury_db=$Faithbench_PATH/pilot_bge_small.sqlite --user_db $Faithbench_PATH/unified_users.sqlite.pilot  --port=8000
python3 $MERCURY_PATH/server.py --mercury_db=$Faithbench_PATH/pilot_openai_small.sqlite --user_db $Faithbench_PATH/unified_users.sqlite.pilot  --port=8001
#python3 $MERCURY_PATH/server.py --mercury_db=$Faithbench_PATH/pilot_all-mpnet-base-v2.sqlite --user_db $Faithbench_PATH/unified_users.sqlite.pilot  --port=8002

