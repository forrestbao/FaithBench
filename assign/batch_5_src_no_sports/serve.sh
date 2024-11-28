# Assume mercury ingest is in the path

MERCURY_PATH="." # the root directory of Mercury's repository
SQLITE_PATH="./batch_5_src_no_sports/users_migrated" # the root directory of Faithbench's repository

source /home/forrest/label_env/bin/activate

export SECRET_KEY=$(openssl rand -base64 32)

for batch in {1..9}
do
python3 $MERCURY_PATH/server.py --mercury_db=$SQLITE_PATH/batch_$batch.sqlite --user_db $SQLITE_PATH/unified_users.sqlite.phase1  --port=$(($batch + 3000))  & 
done
