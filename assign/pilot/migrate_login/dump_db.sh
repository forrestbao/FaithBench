MERCURY=/home/forrest/repos/mercury

for embedder in openai_small
do 
     python3 $MERCURY/database.py pilot_$embedder.sqlite --dump_file pilot_$embedder\_annotation.json
done 

