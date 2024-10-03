gcloud compute scp leaderboard-annotation:~/mercury/Faithbench/pilot_all-mpnet-base-v2.sqlite .  --project "ml-training-382218" --zone "us-central1-c"
#gcloud compute scp leaderboard-annotation:~/mercury_latest/batch_1_2/*.sqlite .  --project "ml-training-382218" --zone "us-central1-c"
gcloud compute scp leaderboard-annotation:~/mercury_latest/Faithbench/pilot_bge_small.sqlite .  --project "ml-training-382218" --zone "us-central1-c"
gcloud compute scp leaderboard-annotation:~/mercury_latest/Faithbench/pilot_openai_small.sqlite .  --project "ml-training-382218" --zone "us-central1-c"
