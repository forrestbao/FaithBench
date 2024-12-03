import jsonlines
import json
from sklearn.metrics import balanced_accuracy_score, f1_score

labels = []
preds = []
with open('processed_ragas_claim_level_preds.jsonl') as reader:
    for record in jsonlines.Reader(reader):
            for sent, sent_result in record['results'].items():
                ## Worst-pooling
                # if 'Unwanted' in sent_result['labels'] or 'Questionable' in sent_result['labels'] or 'Benign' in sent_result['labels']:
                #     labels.append(0)
                # else:
                #     labels.append(1)
                ## Best-pooling
                if 'Consistent' in sent_result['labels']:
                    labels.append(1)
                else:
                    labels.append(0)

                if 0 in sent_result['claim_preds']:
                    preds.append(0)
                else:
                    preds.append(1)
detector_results = {
    "ba": round(balanced_accuracy_score(labels, preds)*100,2),
    "f1-macro": round(f1_score(labels, preds, pos_label=1, average="macro")*100,2),
                # "f1-halu": round(f1_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                # "pr-halu": round(precision_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                # 're-halu': round(recall_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                # "f1-cons": round(f1_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2),
                # "pr-cons": round(precision_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2),
                # 're-cons': round(recall_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2)
}
print(json.dumps(detector_results, indent=2))