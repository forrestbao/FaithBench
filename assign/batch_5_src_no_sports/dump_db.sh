MERCURY=/home/amin/clients/mercury

for batch in {1..9}
#for batch in {3..3}
do 
     python3 $MERCURY/database.py results/batch_$batch.sqlite --dump_file results/batch_$batch\_annotation.json
done 

