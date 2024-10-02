MERCURY=/home/forrest/repos/mercury

for embedder in bge_small openai_small "all-mpnet-base-v2"
do 
     python3 $MERCURY/database.py pilot_$embedder.sqlite --dump_file pilot_$embedder\_annotation.json
done 

