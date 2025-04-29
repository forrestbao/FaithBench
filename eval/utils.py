import json
# import re
import os
from typing import Literal 
import krippendorff
# import nltk
import jsonlines
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from config import REF_FILES, RESULT_PATH, DETECTORS, SUMMARY_SENT_FILE
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/assign')
from annotation_config import skip_samples, annotator_list, three_annotators, exclude_batch


def process_result_files():
    result_files = []
    skip_sample_ids = {}
    selected_annotators = {}
    num_annotators = {}

    for batch_id in range(1,16+1):
        if batch_id in exclude_batch:
            continue
        file_name = os.path.join(RESULT_PATH, f"batch_{batch_id}_annotation.json")
        result_files.append(file_name)
        if batch_id in skip_samples:
            skip_sample_ids[file_name] = [str(s_id) for s_id in skip_samples[batch_id]]
        if batch_id in annotator_list:
            selected_annotators[file_name] = annotator_list[batch_id]
        if batch_id in three_annotators:
            num_annotators[file_name] = 3
        else:
            num_annotators[file_name] = 2
    return result_files, skip_sample_ids, selected_annotators, num_annotators

def union_lists(lst_of_lists):
    # find the union of two or more lists
    if not lst_of_lists:
        return []
    result = lst_of_lists[0]
    for lst in lst_of_lists[1:]:
        result.extend(lst)
    return result

def find_annotation_sentence(annotation_spans, sentences_with_indices):
    # Check which sentence each annotation span belongs to
    annotation_to_sentence = {sent: [] for sent, _, _ in sentences_with_indices}
    for annotation, start, end in annotation_spans:
        for sent, sent_start, sent_end in sentences_with_indices:
            intersection_start = max(start, sent_start)
            intersection_end = min(end, sent_end)
            if intersection_end >= intersection_start:
                intersection_length = intersection_end - intersection_start + 1
                if intersection_length > 3: # in some case, extra white space or letters might be selected. so set >3 as threshold for real annotation
                    annotation_to_sentence[sent].extend(annotation)
    for sent in annotation_to_sentence:
        if len(annotation_to_sentence[sent]) == 0: # no annotation -> consistent
            annotation_to_sentence[sent] = ['Consistent']
        else:
            annotation_to_sentence[sent] = list(set(annotation_to_sentence[sent]))
    return annotation_to_sentence


def best_pooling_01label(labels, halu_labels = ['Unwanted', 'Questionable', 'Benign']):
    if 'Consistent' in labels:
        return 1
    else:
        return 0

def worst_pooling_01label(labels, halu_labels = ['Unwanted', 'Questionable', 'Benign']):
    for l in halu_labels:
        if l in labels:
            return 0
    else:
        return 1

def worst_pooling_4label(labels):
    if "Unwanted" in labels:
        label = "Unwanted"
    elif 'Questionable' in labels:
        label = "Questionable"
    elif "Benign" in labels:
        label = "Benign"
    else:
        label = "Consistent"
    return label

def best_pooling_4label(labels):
    if "Consistent" in labels:
        label = "Consistent"
    elif "Benign" in labels:
        label = "Benign"
    elif 'Questionable' in labels:
        label = "Questionable"
    else:
        label = "Unwanted"
    return label

# Read Annotation
def read_annotation(file_path, skip_sample_ids=[], skip_meta_sample_ids=[]):
    split_text_to_sentences_with_indices = {}
    with open(SUMMARY_SENT_FILE) as reader:
        for record in jsonlines.Reader(reader):
            # record = record[0]
            split_text_to_sentences_with_indices[record['meta_id']] = record['sent_list']
    annotators_records = {} # annotator: {sample1: [label], sample2: [label], ...}
    annotators_sents_records = {} # annotator: {sample1_sent1: [label], sample1_sent2: [label], ...}
    sent_level_labels = {}
    data = json.load(open(file_path))
    # meta_sample_ids = []
    sample_ids = []
    annotators = set([a['annotator'] if not a['annotator_name'] else a['annotator_name'].split()[0].lower() for sample in data for a in sample['annotations']])
    # print(annotators)
    for sample in data:
        annotations = sample['annotations']
        # skip certain samples
        sample_id = sample['sample_id'] # the id in batch
        if skip_sample_ids and sample_id in skip_sample_ids:
            continue
        meta_sample_id = sample['meta_sample_id'] # global id
        if skip_meta_sample_ids and meta_sample_id in skip_meta_sample_ids:
            continue
        # meta_sample_ids.append(meta_sample_id)
        summary = sample['summary']
        summary_sents = split_text_to_sentences_with_indices[meta_sample_id]
        # print(summary_sents)

        sample_ids.append(sample_id)
        annotation_spans = {}
        for annotation in annotations:
            annotator = annotation['annotator'] if not annotation['annotator_name'] else annotation['annotator_name'].split()[0].lower()
            if 'summary_start' not in annotation or len(annotation['label']) < 1: # no selected summary span / no label assigned -> invalid annotation, ignore
                continue
            if annotator not in annotators_records:
                annotators_records[annotator] = {}
            # if meta_sample_id not in annotators_records[annotator]:
            #     annotators_records[annotator][meta_sample_id] = []
            # annotators_records[annotator][meta_sample_id] += annotation['label']
            if sample_id not in annotators_records[annotator]:
                annotators_records[annotator][sample_id] = []
            annotators_records[annotator][sample_id] += annotation['label']

            if annotator not in annotation_spans:
                annotation_spans[annotator] = []
            annotation_spans[annotator].append([annotation['label'], annotation['summary_start'], annotation['summary_end']])
            
        for annotator in annotators:
            if annotator not in annotators_sents_records:
                annotators_sents_records[annotator] = {}
            # if annotator in annotation_spans:
            annotators_sents_records[annotator].update(find_annotation_sentence(annotation_spans.get(annotator, []), summary_sents))
        sent_level_labels[meta_sample_id] = find_annotation_sentence(union_lists(list(annotation_spans.values())), summary_sents)
    
    return sample_ids, annotators_records, annotators_sents_records, sent_level_labels

# Inter-annotator agreement
def compute_span_level_interannotator_agreement(
        file_path, 
        label_map, 
        level_of_measurement='interval', 
        selected_annotators = None, 
        skip_sample_ids=[], 
        skip_meta_sample_ids=[]
    ):
    _, _, annotators_sents_records, _ = read_annotation(file_path, skip_sample_ids=skip_sample_ids, skip_meta_sample_ids=skip_meta_sample_ids)
    # print(sample_ids)
    print('compute span level interannotator agreement')
    print(file_path)
    
    if selected_annotators:
        annotators = []
        for annotator in selected_annotators:
            if annotator not in annotators_sents_records:
                print(f"No records from annotator {annotator}")
            else:
                annotators.append(annotator)
    else:
        annotators = list(annotators_sents_records.keys())
    if not annotators:
        return
    print('annotator for span level agreement computation:', annotators)
    results= {annotator: [] for annotator in annotators} # annotator 1: [label for sample1_sent1, label for sample1_sent2 ....] # either consistent or hallucinated
    # those are annotated examples
    for annotator in annotators:
        print(len(annotators_sents_records[annotator].values()))
        for sent in annotators_sents_records[annotator]:
            # consider the worst label
            sent_label = annotators_sents_records[annotator][sent]
            if 'Unwanted' in sent_label:
                label = label_map['unwanted']
            elif label_map['questionable'] > label_map['benign']:
                if 'Benign' in sent_label:
                    label = label_map['benign']
                else:
                    label = label_map['questionable']
            else:   
                if 'Questionable' in sent_label:
                    label = label_map['questionable']
                else:
                    label = label_map['benign']

            results[annotator].append(label)
    print(results)
    # print(sample_ids)
    # annotation_labels = list(results.values())
    disagree_sent_ids = []
    for i in range(len(results[annotator])):
        if len(set([labels[i] for _, labels in results.items()])) > 1: 
            disagree_sent_ids.append(i)
    print('disagreed sent ids:', disagree_sent_ids)
    for annotator in annotators:
        print(f"{annotator} {[results[annotator][i] for i in disagree_sent_ids]}")

    print(f"{level_of_measurement} Krippendorff\'s alpha for label map\n{label_map}")
    value_domain= sorted(list(set([l for l in label_map.values() if not np.isnan(l)])))
    # if all annotators only have the same single type of label, the alpha will be nan
    # to deal with this issue, manually add an extra annotion for the other label for all annotators
    all_annotations = []
    for annotator in results:
        anno = np.array(results[annotator])
        all_annotations.extend(anno[np.logical_not(np.isnan(anno))])
    if len(set(all_annotations)) == 1:
        other_labels = value_domain.copy()
        other_labels.remove(list(set(all_annotations))[0])
        for annotator in results:
            results[annotator].append(other_labels[0])
    
    agreement = krippendorff.alpha(np.array(list(results.values()), dtype=np.dtype(float)), level_of_measurement=level_of_measurement, value_domain=value_domain)
    print(round(agreement,3))
    return agreement

def compute_interannotator_agreement(
        file_path, 
        label_map, 
        level_of_measurement='interval', 
        selected_annotators = None, 
        skip_sample_ids=[], 
        skip_meta_sample_ids=[]
    ):
    sample_ids, annotators_records, _, _ = read_annotation(file_path, skip_sample_ids=skip_sample_ids, skip_meta_sample_ids=skip_meta_sample_ids)
    sample_ids = sorted(sample_ids)
    # print(sample_ids)
    # print(file_path)
    
    if selected_annotators:
        annotators = []
        for annotator in selected_annotators:
            if annotator not in annotators_records:
                print(f"No records from annotator {annotator}")
            else:
                annotators.append(annotator)
    else:
        annotators = list(annotators_records.keys())
    if not annotators:
        return
    print('annotator for agreement computation:', annotators)
    results= {annotator: [] for annotator in annotators} # annotator 1: [label for sample 1, label for sample 2 ....] # either consistent or hallucinated
    # those are annotated examples
    for sample_id in sample_ids:
        for annotator in annotators:
            if sample_id not in annotators_records[annotator]:
                label = label_map['consistent']
            else:
                # consider the worst label
                sample_label = set(annotators_records[annotator][sample_id])
                if 'Unwanted' in sample_label:
                    label = label_map['unwanted']
                elif label_map['questionable'] > label_map['benign']:
                    if 'Benign' in sample_label:
                        label = label_map['benign']
                    else:
                        label = label_map['questionable']
                else:   
                    if 'Questionable' in sample_label:
                        label = label_map['questionable']
                    else:
                        label = label_map['benign']

            results[annotator].append(label)

    # print(sample_ids)
    annotation_labels = list(results.values())
    disagree_sample_ids = []
    for sample_id in sample_ids:
        sample_annotations = [annotation_label[sample_id] for annotation_label in annotation_labels]
        if len(set(sample_annotations)) > 1:
            # print('different annotation for sample id:', sample_id + 1)
            disagree_sample_ids.append(sample_id)
    print('disagreed sample ids in batch:', [sid+1 for sid in disagree_sample_ids])
    for annotator in annotators:
        print(f"{annotator} {[results[annotator][i] for i in disagree_sample_ids]}")

    print(f"{level_of_measurement} Krippendorff\'s alpha for label map\n{label_map}")
    value_domain= sorted(list(set([l for l in label_map.values() if not np.isnan(l)])))
    # if all annotators only have the same single type of label, the alpha will be nan
    # to deal with this issue, manually add an extra annotion for the other label for all annotators
    all_annotations = []
    for annotator in results:
        anno = np.array(results[annotator])
        all_annotations.extend(anno[np.logical_not(np.isnan(anno))])
    if len(set(all_annotations)) == 1:
        other_labels = value_domain.copy()
        other_labels.remove(list(set(all_annotations))[0])
        for annotator in results:
            results[annotator].append(other_labels[0])
    
    agreement = krippendorff.alpha(np.array(list(results.values()), dtype=np.dtype(float)), level_of_measurement=level_of_measurement, value_domain=value_domain)
    print(round(agreement,3))
    return agreement


class DetectorEvaluator():
    def __init__(self, result_files, sample_pooling, halu_labels=['Unwanted', 'Questionable'], skip_sample_ids={}, skip_meta_sample_ids=[], selected_annotators={}, num_annotators={}):
        '''
        result_files: the list of files to process
        sample_pooling: sample pooling method
        skip_sample_ids: batch sample id of samples to be skipped when computing the results
            {filename: [sample ids to skip in the file]}
        skip_meta_sample_ids: meta samplfhalu) id of samples to be skipped
        selected_annotators: selected annotator for each file
            {filename: [selected annotators]}
        num_annotators: number of annotators for each file
        '''
        self.detectors = DETECTORS
        self.predictions = {detector: [] for detector in ['human', 'human_4labels'] + list(self.detectors.values())}
        self.result_files = result_files
        self.skip_sample_ids = skip_sample_ids
        self.skip_meta_sample_ids = skip_meta_sample_ids
        self.selected_annotators = selected_annotators
        self.halu_labels = halu_labels
        self.num_annotators = num_annotators
        assert sample_pooling in ['worst-pooling', 'best-pooling'], "only `worst-pooling` and `best-pooling` are supported"
        self.sample_pooling = sample_pooling
        # self.human_mapped_labels = {label: 0 if label in self.halu_labels else 1
        #                        for label in ['Consistent', 'Benign', 'Questionable', 'Unwanted']}
        
    def process_results(self):
        self.batch_predictions = {file_path: {'preds': {detector: [] for detector in list(self.detectors.values())}, 'avg_source_len': 0, 'avg_summary_len': 0}  for file_path in self.result_files}
        ref_df_list = []
        for fp in REF_FILES:
            df = pd.read_csv(fp)
            ref_df_list.append(df)
        ref_data = pd.concat(ref_df_list, ignore_index=True)
        for file_path in self.result_files:
            data = json.load(open(file_path))
            selected_annotators = None
            num_annotator = self.num_annotators[file_path]
            if file_path in self.selected_annotators:
                selected_annotators = self.selected_annotators[file_path]
        
            sample_count = 0
            for sample in data:
                sample_id = sample['sample_id']
                if file_path in self.skip_sample_ids and str(sample_id) in self.skip_sample_ids[file_path]:
                    continue
                meta_sample_id = sample['meta_sample_id']
                if meta_sample_id in self.skip_meta_sample_ids:
                    continue
                sample_count += 1
                source = sample['source']
                summary = sample['summary']
                self.batch_predictions[file_path]['avg_source_len'] += len(source.split())
                self.batch_predictions[file_path]['avg_summary_len'] += len(summary.split())
                llm = sample['meta_model']
                annotations = sample['annotations']
                sample_annotations = []
                occurred_annotators = set()
                for annotation in annotations:
                    annotator = annotation['annotator'] if not annotation['annotator_name'] else annotation['annotator_name'].split()[0].lower()
                    if selected_annotators:
                        if annotator in selected_annotators:
                            sample_annotations.extend(annotation['label'])
                            occurred_annotators.add(annotator)
                    else:
                        sample_annotations.extend(annotation['label'])
                        occurred_annotators.add(annotator)
                    
                if len(occurred_annotators) < num_annotator:
                    sample_annotations.extend(['Consistent']* (num_annotator - len(occurred_annotators)))
                
                sample_annotations = set(sample_annotations)
                # human annotation
                if self.sample_pooling == 'worst-pooling':
                    sample_pred = worst_pooling_4label(sample_annotations)
                
                else:
                    sample_pred = best_pooling_4label(sample_annotations)
                    
                if sample_pred in self.halu_labels:
                    human_label = 0
                else:
                    human_label = 1
                    
                self.predictions['human'].append(human_label)
                self.predictions['human_4labels'].append(sample_pred)
                # row = ref_data.loc[(ref_data['source'] == source) & (ref_data['model'] == llm)]
                row = ref_data.loc[meta_sample_id]
                # if row.empty:
                #     row = ref_data.loc[(ref_data['source'] == '\"'+source+'\"') & (ref_data['model'] == llm)]
                # if row.empty:
                if row['model'] != llm:
                    print(source)
                    print(llm)
                # print(row)
                for detector in self.detectors:
                    detector_pred = row[f"{detector}"].astype(int)
                    if 'HHEM' in detector or 'alignscore' in detector.lower():
                        detector_pred = 0 if row[f"{detector}"].astype(float) < 0.5 else 1
                    elif 'Ragas' in detector or 'Trulens' in detector:
                        detector_pred = 1 if row[f"{detector}"] == 1 else 0
                    elif detector_pred not in [0,1]: # invalid prediction, consider wrong
                        detector_pred = 1 - human_label
                    self.predictions[self.detectors[detector]].append(detector_pred)
                    self.batch_predictions[file_path]['preds'][self.detectors[detector]].append(detector_pred)

            self.batch_predictions[file_path]['avg_source_len'] /= sample_count
            self.batch_predictions[file_path]['avg_summary_len'] /= sample_count

        self.pred_df = pd.DataFrame(self.predictions)
        print(self.pred_df.shape)

    def compute_correlation(self, correlation_method):
        return self.pred_df.loc[:, self.pred_df.columns!='human_4labels'].corr(method=correlation_method).round(2)
    
    def compute_performance(self):
        self.performance_results = {}
        for detector in list(self.detectors.values()):
            detector_results = {
                "ba": round(balanced_accuracy_score(self.pred_df['human'], self.pred_df[detector])*100,2),
                "f1-macro": round(f1_score(self.pred_df['human'], self.pred_df[detector], pos_label=1, average="macro")*100,2),
                "f1-halu": round(f1_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                # "pr-halu": round(precision_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                # 're-halu': round(recall_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                # "f1-cons": round(f1_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2),
                # "pr-cons": round(precision_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2),
                # 're-cons': round(recall_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2)
            }
            self.performance_results[detector] = detector_results
        return pd.DataFrame.from_dict(self.performance_results, orient='index')
    
    def get_error_distribution(self):
        # Initialize a dictionary to hold error counts for each model and each label category
        self.error_distribution = {model: {'Unwanted': 0, 'Questionable': 0, 'Benign': 0, 'Consistent': 0}
                              for model in self.predictions if 'human' not in model}
        
        # Extract the human ground truth labels
        human_01labels = self.predictions['human']
        human_4labels = self.predictions['human_4labels']
        
        # Iterate over each model's predictions (excluding the 'human' entry)
        for model, model_predictions in self.predictions.items():
            if 'human' in model:
                continue
            # Compare each prediction with the mapped ground truth
            for idx, pred in enumerate(model_predictions):
                if human_01labels[idx] != pred:
                    self.error_distribution[model][human_4labels[idx]] += 1
        
        for model in self.error_distribution:
            total = sum(list(self.error_distribution[model].values()))
            for l in self.error_distribution[model]:
                self.error_distribution[model][l] = round(self.error_distribution[model][l]/total*100, 2)

        # return self.error_distribution
        # colors = [mcolors.to_rgba(c, alpha=0.7) for c in [plt.cm.tab10(2), plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(3)]]  # Red, Orange, Blue, Green,  for each category respectively
        colors = [mcolors.to_rgba(c, alpha=0.7) for c in [plt.cm.tab10(3), plt.cm.tab10(1), plt.cm.tab10(0), plt.cm.tab10(2)]]
        
        error_distribution_df = pd.DataFrame.from_dict(self.error_distribution, orient='index')
        
        # Plotting the stacked bar plot
        ax = error_distribution_df.plot(kind='bar', stacked=True, figsize=(16,6), color=colors)
        # fig, ax = plt.subplots(figsize=(12, 8))
        for container in ax.containers:
            # ax.bar_label(container, fmt='%.2f', label_type='center', rotation=45, color='black', fontsize=9, padding=1)
            labels = [f"{v:.2f}" if v >= 4 else "" for v in container.datavalues]
            ax.bar_label(container, labels=labels, label_type='center', rotation=20, color='black', fontsize=13, padding=0)

        # Add labels and title
        # plt.xlabel('Model')
        plt.ylabel('Distribution of labels (%)', fontsize=15)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=15, title_fontsize='small', frameon=True, ncol=4)
        plt.xticks(rotation=20, ha='right', fontsize=15)
        plt.tight_layout()
        plt.show()

    def disagree_vs_length(self):
        source_lens = []
        summary_lens = []
        disagreement_count = []
        
        for _, data in self.batch_predictions.items():
            preds = data['preds']
            num_samples = len(next(iter(preds.values())))
            batch_disagreement = []
            
            # Loop through each sample (index)
            for i in range(num_samples):
                # Get all predictions for this sample from all models
                current_predictions = [preds[model][i] for model in preds]
                
                # Check if there is disagreement by comparing the set of predictions
                if len(set(current_predictions)) > 1:
                    batch_disagreement.append(True)  # There is a disagreement
                else:
                    batch_disagreement.append(False)  # No disagreement
                    
            disagreement_count.append(sum(batch_disagreement))
            source_lens.append(data['avg_source_len'])
            summary_lens.append(data['avg_summary_len'])

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        # Use a color map ("Greys") to better represent disagreement intensity with dark colors for larger counts
        cmap = plt.cm.get_cmap("Wistia")
        
        # Plot source length vs disagreement with color representing disagreement count
        scatter1 = axs[0].scatter(source_lens, disagreement_count, c=disagreement_count, cmap=cmap, s=60, alpha=0.9)
        axs[0].set_xlabel("Source Length (words)", fontsize=9)
        axs[0].set_ylabel("Disagreement Number", fontsize=9)
        axs[0].set_title('Source Length vs Disagreement', fontsize=10)
        axs[0].grid(True, linestyle='--', alpha=0.5)
        
        # Add disagreement count labels to every data point with rotation and merging close points
        for i, (src_len, disag) in enumerate(zip(source_lens, disagreement_count)):
            label_placed = False
            for j in range(i):
                # Check if another label is close
                if abs(src_len - source_lens[j]) < 30 and abs(disag - disagreement_count[j]) < 2:
                    label_placed = True
                    break
            if not label_placed:
                axs[0].text(src_len, disag, f'{disag}', fontsize=7, ha='right', rotation=45, color='black')

        # Plot summary length vs disagreement with color representing disagreement count
        scatter2 = axs[1].scatter(summary_lens, disagreement_count, c=disagreement_count, cmap=cmap, s=50, alpha=0.7)
        axs[1].set_xlabel("Summary Length (words)", fontsize=9)
        axs[1].set_ylabel("Disagreement Number", fontsize=9)
        axs[1].set_title('Summary Length vs Disagreement', fontsize=10)
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        # Add disagreement count labels to every data point with rotation and merging close points
        for i, (sum_len, disag) in enumerate(zip(summary_lens, disagreement_count)):
            label_placed = False
            for j in range(i):
                # Check if another label is close
                if abs(sum_len - summary_lens[j]) < 15 and abs(disag - disagreement_count[j]) < 2:
                    label_placed = True
                    break
            if not label_placed:
                axs[1].text(sum_len, disag, f'{disag}', fontsize=7, ha='right', rotation=45, color='black')

        # Adjust layout for better spacing
        plt.tight_layout()

        # Add a color bar to represent the disagreement count
        cbar = fig.colorbar(scatter1, ax=axs, orientation='vertical', fraction=0.02, pad=0.05)
        cbar.set_label('Disagreement Count', fontsize=9)

        # Show the plot
        plt.show()
        
    def disagree_vs_model(self):
        num_model_vs_count = {}
        
        for _, data in self.batch_predictions.items():
            preds = data['preds']
            num_samples = len(next(iter(preds.values())))
            batch_disagreement = []
            
            # Loop through each sample (index)
            for i in range(num_samples):
                # Get all predictions for this sample from all models
                current_predictions = [preds[model][i] for model in preds]
                
                # Check if there is disagreement by comparing the set of predictions
                if len(set(current_predictions)) > 1:
                    num_disagreed_model = min(list(dict(Counter(current_predictions)).values())) # There is a disagreement
                    if not num_disagreed_model in num_model_vs_count:
                        num_model_vs_count[num_disagreed_model] = 0
                    num_model_vs_count[num_disagreed_model] += 1
            
        x_labels = list(num_model_vs_count.keys())
        y_values = list(num_model_vs_count.values())

        # Create the bar plot
        plt.figure(figsize=(4, 3))
        bars = plt.bar(x_labels, y_values, color='skyblue')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}', ha='center', fontsize=6)


        # Adding labels and title
        plt.xlabel('Number of Disagreed Models', fontsize=8)
        plt.ylabel('Disagreement Count', fontsize=8)
        # plt.title('Disagreement Count by Model', fontsize=10)
        # plt.xticks(rotation=45, ha='right')

        # Display the plot
        plt.tight_layout()
        plt.show()
                    

class SentLevelDetectorEvaluator(DetectorEvaluator):
    def __init__(self, result_files, sample_pooling, halu_labels=['Unwanted', 'Questionable'], skip_sample_ids={}, skip_meta_sample_ids=[], selected_annotators={}, num_annotators={}):
        super(self.__class__, self).__init__(result_files, sample_pooling, halu_labels, skip_sample_ids, skip_meta_sample_ids, selected_annotators, num_annotators)

        # self.detectors = DETECTORS
        # self.predictions = {detector: [] for detector in ['human', 'human_4labels'] + list(self.detectors.values())}
        # self.result_files = result_files
        # self.skip_sample_ids = skip_sample_ids
        # self.skip_meta_sample_ids = skip_meta_sample_ids
        # self.selected_annotators = selected_annotators
        # self.halu_labels = halu_labels
        # self.num_annotators = num_annotators
        # assert sample_pooling in ['worst-pooling', 'best-pooling'], "only `worst-pooling` and `best-pooling` are supported"
        # self.sample_pooling = sample_pooling
        # self.human_mapped_labels = {label: 0 if label in self.halu_labels else 1
        #                        for label in ['Consistent', 'Benign', 'Questionable', 'Unwanted']}
    
    def process_results(self):
        self.predictions = {detector: [] for detector in ['human', 'human_4labels'] + list(self.detectors.values())}
        for detector in ['human', 'human_4labels']:
            with open('sent_level_results/detectors_sent_level_preds_merged.json') as f:
                data = json.load(f)
                for meta_id in data:
                    record = data[meta_id]
                    for _, results in record.items():
                        if detector == 'human':
                            if self.sample_pooling == 'best-pooling':
                                self.predictions[detector].append(best_pooling_01label(results['labels']))
                            else:
                                self.predictions[detector].append(worst_pooling_01label(results['labels']))
                        elif detector == 'human_4labels':
                            if self.sample_pooling == 'best-pooling':
                                self.predictions[detector].append(best_pooling_4label(results['labels']))
                            else:
                                self.predictions[detector].append(worst_pooling_4label(results['labels']))
                            
        for detector in self.detectors:
            # print(detector)
            if 'ragas' in detector.lower():
                with open('sent_level_results/processed_ragas_claim_level_preds.jsonl') as reader:
                    for record in jsonlines.Reader(reader):
                        for _, sent_result in record['results'].items():
                            if len(sent_result['claims']) < 1: # no prediction
                                self.predictions[self.detectors[detector]].append(1)
                            elif 0 in sent_result['claim_preds']:
                                self.predictions[self.detectors[detector]].append(0)
                            else:
                                self.predictions[self.detectors[detector]].append(1)
            elif 'trulens' in detector.lower():
                with open('sent_level_results/processed_trulens_claim_level_preds.jsonl') as reader:
                    for record in jsonlines.Reader(reader):
                            for _, sent_result in record['results'].items():
                                if len(sent_result['claims']) < 1: # no prediction
                                    self.predictions[self.detectors[detector]].append(1)
                                elif np.mean(sent_result['claim_preds']) < 1:
                                    self.predictions[self.detectors[detector]].append(0)
                                else:
                                    self.predictions[self.detectors[detector]].append(1)
            else:
                with open('sent_level_results/detectors_sent_level_preds_merged.json') as f:
                    data = json.load(f)
                    for meta_id in data:
                        record = data[meta_id]
                        for _, results in record.items():
                            if detector not in results:
                                continue
                            if results[detector] is not None:
                                self.predictions[self.detectors[detector]].append(int(results[detector] > 0.5))
                            else: # prediction is None. may occur for trueteacher/truenli
                                if self.sample_pooling == 'best-pooling':
                                    self.predictions[self.detectors[detector]].append(1-best_pooling_01label(results['labels']))
                                else:
                                    self.predictions[self.detectors[detector]].append(1-worst_pooling_01label(results['labels']))
        for detector, results in self.predictions.items():
            print(detector, len(results))                
        self.pred_df = pd.DataFrame(self.predictions)
        # return self.predictions


class HaluEvaluator():
    def __init__(self, result_files, sample_pooling, halu_labels=['Unwanted', 'Questionable'], skip_sample_ids={}, skip_meta_sample_ids=[], selected_annotators={}, num_annotators={}):
        '''
        result_files: the list of files to process
        skip_sample_ids: batch sample id of samples to be skipped when computing the results
            {filename: [sample ids to skip in the file]}
        skip_meta_sample_ids: meta sample id of samples to be skipped
        selected_annotators: selected annotator for each file
            {filename: [selected annotators]}
        num_annotators: number of annotators for each file
        '''
        self.result_files = result_files
        self.skip_sample_ids = skip_sample_ids
        self.skip_meta_sample_ids = skip_meta_sample_ids
        self.selected_annotators = selected_annotators
        self.num_annotators = num_annotators
        self.model_map = {
            "openai/GPT-3.5-Turbo": "GPT-3.5-Turbo",
            "openai/gpt-4o": "GPT-4o",
            "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
            "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini",
            "cohere/command-r-08-2024": "Command-R",
            "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
            "meta-llama/Meta-Llama-3.1-70B-Instruct":"Llama-3.1-70B",
            "google/gemini-1.5-flash-001": "Gemini-1.5-Flash",
            "Anthropic/claude-3-5-sonnet-20240620": "Claude-3.5-Sonnet",
            "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B"
        }
        self.halu_labels=halu_labels
        assert sample_pooling in ['worst-pooling', 'best-pooling'], "only `worst-pooling` and `best-pooling` are supported"
        self.sample_pooling = sample_pooling
        

    def process_results(self):
        self.model_preds = {model: {'sample_labels': [], 'avg_annotations': {l:0 for l in ['Unwanted','Questionable', 'Benign', 'Consistent']}} for model in self.model_map}
        self.batch_model_len_preds = {file_path:{model: {'avg_annotations': {l:0 for l in ['Unwanted','Questionable', 'Benign']}, 'sample_labels':[], 'avg_source_len': 0, 'avg_summary_len': 0} for model in self.model_map} for file_path in self.result_files}
        # avg_annotation = num of one label / num of annotators
        # i.e., if one annotator made 4 unwanted annotations to a sample and the other annotator marked 2 benign labels, the labels for this sample are 2 unwanted and 1 benign
        total_sample_count = 0
        for file_path in self.result_files:
            # print("file_path:", file_path)
            data = json.load(open(file_path))
            selected_annotators = None
            num_annotator = self.num_annotators[file_path]
            if file_path in self.selected_annotators:
                selected_annotators = self.selected_annotators[file_path]
        
            sample_count = 0
            for sample in data:
                sample_id = sample['sample_id']
                # print("sample_id:", sample_id)
                if file_path in self.skip_sample_ids and str(sample_id) in self.skip_sample_ids[file_path]:
                    # print('skip')
                    continue
                meta_sample_id = sample['meta_sample_id']
                if meta_sample_id in self.skip_meta_sample_ids:
                    continue
                sample_count += 1
                model_name = sample['meta_model']
                annotations = sample['annotations']
                # self.model_preds[model_name]['source_len'].append(len(sample['source'].split()))
                # self.model_preds[model_name]['summary_len'].append(len(sample['summary'].split()))
                self.batch_model_len_preds[file_path][model_name]['avg_source_len'] += len(sample['source'].split())
                self.batch_model_len_preds[file_path][model_name]['avg_summary_len'] += len(sample['summary'].split())
                
                
                sample_annotations = []
                occurred_annotators = set()
                for annotation in annotations:
                    annotator = annotation['annotator'] if not annotation['annotator_name'] else annotation['annotator_name'].split()[0].lower()
                    if selected_annotators:
                        if annotator in selected_annotators:
                            sample_annotations.extend(annotation['label'])
                            occurred_annotators.add(annotator)
                    else:
                        sample_annotations.extend(annotation['label'])
                        occurred_annotators.add(annotator)

                annotation_counter = dict(Counter(sample_annotations))
                for l in ['Unwanted','Questionable', 'Benign']:
                    if l in annotation_counter:
                        self.model_preds[model_name]['avg_annotations'][l] += annotation_counter[l] / num_annotator
                        self.batch_model_len_preds[file_path][model_name]['avg_annotations'][l] += annotation_counter[l] / num_annotator
                
                if len(occurred_annotators) < num_annotator:
                    self.model_preds[model_name]['avg_annotations']['Consistent'] += (num_annotator - len(occurred_annotators)) / num_annotator
                    sample_annotations.extend(['Consistent']* (num_annotator - len(occurred_annotators)))
                sample_annotations = set(sample_annotations)
                # human annotation
                if self.sample_pooling == 'worst-pooling':
                    sample_pred = worst_pooling_4label(sample_annotations)
                
                else:
                    sample_pred = best_pooling_4label(sample_annotations)
                    
                self.model_preds[model_name]['sample_labels'].append(sample_pred)
                self.batch_model_len_preds[file_path][model_name]['sample_labels'].append(sample_pred)
    
            for model_name in self.model_map:
                self.batch_model_len_preds[file_path][model_name]['avg_source_len'] /= (sample_count//len(self.model_map))
                self.batch_model_len_preds[file_path][model_name]['avg_summary_len'] /= (sample_count//len(self.model_map))
            total_sample_count += sample_count
        print("total_sample_count", total_sample_count)

    def compute_halu_rate(self):
        self.model_results = {}
        for model in self.model_preds:
            preds = self.model_preds[model]['sample_labels']
            # print(f'number of records for {model}:',len(preds))
            halu_count = 0
            for pred in preds:
                if pred in self.halu_labels:
                    halu_count += 1
            self.model_results[self.model_map[model]] = round((halu_count/len(preds))*100,2)
        self.model_results = {k: v for k, v in sorted(self.model_results.items(), key=lambda item: item[1])}
        return self.model_results
    
    def get_sample_dist(self):
        self.model_halu_dist = {}
        for model in self.model_preds:
            self.model_halu_dist[model] = {}
            
            # print(preds, sum(list(preds.values())))
            predictions = self.model_preds[model]['sample_labels']
            sample_label_count = dict(Counter(predictions))
            for l in ['Unwanted','Questionable','Benign','Consistent']:
                self.model_halu_dist[model][l] = round(sample_label_count[l]/(sum(list(sample_label_count.values())))*100, 2)
            

        halu_dist_df = pd.DataFrame.from_dict(self.model_halu_dist, orient='index') 
        # Set the model names as index for easier plotting
        halu_dist_df.index = halu_dist_df.index.map(lambda x: self.model_map[x])
        # halu_dist_df = halu_dist_df.sort_values(by='Unwanted', ascending=True)
        halu_dist_df = halu_dist_df.reindex(['GPT-4o', 'GPT-3.5-Turbo', "Llama-3.1-70B", "Claude-3.5-Sonnet", "Gemini-1.5-Flash", 'Llama-3.1-8B', "Qwen2.5-7B", "Command-R", "Mistral-7B", "Phi-3-mini"])
        colors = [mcolors.to_rgba(c, alpha=0.7) for c in [plt.cm.tab10(3), plt.cm.tab10(1), plt.cm.tab10(0), plt.cm.tab10(2)]]
        
        # Plot the stacked bar chart
        ax = halu_dist_df.plot(kind='bar', stacked=True, figsize=(12, 3), color=colors)
        # Add numbers on the stacks
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='center', rotation=10, color='black', fontsize=10, padding=1)

        # Add labels and title
        # plt.xlabel('Model')
        plt.ylabel('Distribution of labels (%)', fontsize=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=12, title_fontsize='small', frameon=True, ncol=4)
        plt.xticks(rotation=10, ha='right', fontsize=12)
        plt.tight_layout()
        plt.show()

    def get_annotation_dist(self):
        self.model_anno_dist = {}
        for model in self.model_preds:
            self.model_anno_dist[model] = {}
            
            # print(preds, sum(list(preds.values())))
            for l in ['Unwanted','Questionable','Benign']:#,'Consistent']:
                preds = self.model_preds[model]['avg_annotations']
                self.model_anno_dist[model][l] = round(preds[l]/(sum([preds[lb] for lb in ['Unwanted','Questionable','Benign']]))*100, 2)
                
        halu_dist_df = pd.DataFrame.from_dict(self.model_anno_dist, orient='index') 
        # print(halu_dist_df)
        # Set the model names as index for easier plotting
        halu_dist_df.index = halu_dist_df.index.map(lambda x: self.model_map[x])
        # halu_dist_df = halu_dist_df.sort_values(by='Unwanted', ascending=True)
        halu_dist_df = halu_dist_df.reindex(['GPT-4o', 'GPT-3.5-Turbo', "Llama-3.1-70B", "Claude-3.5-Sonnet", "Gemini-1.5-Flash", 'Llama-3.1-8B', "Qwen2.5-7B", "Command-R", "Mistral-7B", "Phi-3-mini"])
        colors = [mcolors.to_rgba(c, alpha=0.7) for c in [plt.cm.tab10(3), plt.cm.tab10(1), plt.cm.tab10(0), plt.cm.tab10(2)]]
        
        # Plot the stacked bar chart
        ax = halu_dist_df.plot(kind='bar', stacked=True, figsize=(9, 4), color=colors)
        # Add numbers on the stacks
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='center', rotation=65, color='black', fontsize=12, padding=1)

        # Add labels and title
        # plt.xlabel('Model')
        plt.ylabel('Share of annotations (%)', fontsize=14)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=14, title_fontsize='small', frameon=True, ncol=4)
        plt.xticks(rotation=20, ha='right', fontsize=14)
        plt.tight_layout()
        plt.show()

    def halu_vs_length(self, length_of: Literal['source', 'summary']):
        # Initialize the plot
        plt.figure(figsize=(9, 4))
        
        # Define base colors and line styles for model families with more distinguishable styles
        base_colors = {
            "openai": '#1f77b4',  # Blue
            "Qwen": '#ff7f0e',    # Orange
            "microsoft": '#2ca02c',  # Green
            "cohere": '#d62728',    # Red
            "meta-llama": '#9467bd',  # Purple
            "google": '#8c564b',    # Brown
            "Anthropic": '#e377c2',  # Pink
            "mistralai": '#7f7f7f'   # Gray
        }
        
        line_styles = ['-', '--', '-.', ':']  # Different line styles to distinguish models within the same family
        family_models = {}  # Dictionary to track models within each family
        
        # Loop to count models within each family
        for batch in self.batch_model_len_preds.keys():
            for model in self.batch_model_len_preds[batch].keys():
                family = model.split('/')[0]
                standard_model_name = self.model_map.get(model, model)
                if family not in family_models:
                    family_models[family] = []
                if standard_model_name not in family_models[family]:
                    family_models[family].append(standard_model_name)
        
        # Loop over each batch and model to extract data
        plot_data = {self.model_map[model]: {'halu_rate':[], 'source_len':[], 'summary_len':[]} for model in self.model_map}
        for batch, models_data in self.batch_model_len_preds.items():
            for model, data in models_data.items():
                family = model.split('/')[0]
                standard_model_name = self.model_map.get(model, model)
                
                # Get a distinct color for the family
                color = base_colors.get(family, '#000000')  # Default to black if family not found
                
                # Determine line style: use different styles only for models within the same family
                if len(family_models[family]) > 1:
                    line_style = line_styles[family_models[family].index(standard_model_name) % len(line_styles)]
                else:
                    line_style = '-'
                
                # Extract source length and hallucination counts
                source_length = np.array(data['avg_source_len'])
                summary_length = np.array(data['avg_summary_len'])
                predictions = data['sample_labels']
                hallucinated = [1 if pred in self.halu_labels else 0 for pred in predictions]
                hallucinated = np.array(hallucinated)
                
                plot_data[standard_model_name]['color'] = color
                plot_data[standard_model_name]['line_style'] = line_style
                plot_data[standard_model_name]['halu_rate'].append(round(np.mean(hallucinated) * 100, 2))
                # plot_data[standard_model_name]['source_len'].append(source_length)
                plot_data[standard_model_name][length_of + '_len'].append(source_length if length_of == 'source' else summary_length)
                # Smooth the hallucination rate using a Gaussian filter
                # hallucination_rates_smooth = gaussian_filter1d(hallucination_rates, sigma=5)
        
        for model, data in plot_data.items():
            # x_vals = np.array(data['source_len'])
            x_vals = np.array(data[length_of+'_len'])
            y_vals = np.array(data['halu_rate'])
            sorted_indices = np.argsort(x_vals)
            x_vals = x_vals[sorted_indices]
            y_vals = y_vals[sorted_indices]
            y_smooth = gaussian_filter1d(y_vals, sigma=2)
            plt.plot(x_vals, y_smooth, label=model, color=data['color'], linestyle=data['line_style'], linewidth=1.5)

        # Customize the plot
        plt.xlabel(("Passage" if length_of == 'source' else "Summary")
                    + ' Length (# of words)', fontsize=10)
        plt.ylabel('Hallucination Rate (%)', fontsize=10)
        # plt.title('Hallucination Rate vs. Source Length', fontsize=16)
        # plt.xscale('log')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.get_major_locator().set_params(integer=True)
        
        # Add shared legend at the bottom of the figure
        plt.legend(loc='lower right', fontsize=9, title_fontsize='small', ncol=4, frameon=True) # bbox_to_anchor=(0.5, -0.5), 
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to fit the legend
        plt.show()

    def label_vs_length(self, mode, length_of: Literal['source', 'summary']):
        # Define labels to be plotted
        labels = ['Unwanted', 'Questionable', 'Benign']
        
        # Initialize subplots for each label
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        
        # Ensure axs is iterable by converting it to a list
        axs = list(axs) if isinstance(axs, np.ndarray) else [axs]
        
        # Define base colors and line styles for model families with more distinguishable styles
        base_colors = {
            "openai": '#1f77b4',  # Blue
            "Qwen": '#ff7f0e',    # Orange
            "microsoft": '#2ca02c',  # Green
            "cohere": '#d62728',    # Red
            "meta-llama": '#9467bd',  # Purple
            "google": '#8c564b',    # Brown
            "Anthropic": '#e377c2',  # Pink
            "mistralai": '#7f7f7f'   # Gray
        }
        
        line_styles = ['-', '--', '-.', ':']  # Different line styles to distinguish models within the same family
        family_models = {}  # Dictionary to track models within each family
        
        # Loop to count models within each family
        for batch in self.batch_model_len_preds.keys():
            for model in self.batch_model_len_preds[batch].keys():
                standard_model_name = self.model_map.get(model, model)
                family = model.split('/')[0]
                if family not in family_models:
                    family_models[family] = []
                if standard_model_name not in family_models[family]:
                    family_models[family].append(standard_model_name)
        
        # Initialize data collection for plotting
        plot_data = {label: {} for label in labels}
        
        # Loop over each batch and model to extract data
        for batch, models_data in self.batch_model_len_preds.items():
            for model, data in models_data.items():
                standard_model_name = self.model_map.get(model, model)
                # Extract model family
                family = model.split('/')[0]
                
                # Get a distinct color for the family
                color = base_colors.get(family, '#000000')  # Default to black if family not found
                
                # Determine line style: use different styles only for models within the same family
                if len(family_models[family]) > 1:
                    line_style = line_styles[family_models[family].index(standard_model_name) % len(line_styles)]
                else:
                    line_style = '-'
                
                # Extract source length and label counts
                source_length = data['avg_source_len']
                label_counts = data['avg_annotations']
                total_count = sum(label_counts.values())
                
                # Calculate ratio for each label and store data for plotting
                for label in labels:
                    if standard_model_name not in plot_data[label]:
                        plot_data[label][standard_model_name] = {'x': [], 'y': [], 'color': color, 'line_style': line_style}
                    label_ratio = (label_counts[label] / total_count * 100) if total_count > 0 else 0
                    plot_data[label][standard_model_name]['x'].append(source_length)
                    if mode == 'ratio':
                        plot_data[label][standard_model_name]['y'].append(label_ratio)
                    elif mode == 'count':
                        plot_data[label][standard_model_name]['y'].append(label_counts[label])
                        
        # Plot the data for each label as line plots
        for i, label in enumerate(labels):
            for model, data in plot_data[label].items():
                x_vals = np.array(data['x'])
                y_vals = np.array(data['y'])
                sorted_indices = np.argsort(x_vals)
                x_vals = x_vals[sorted_indices]
                y_vals = y_vals[sorted_indices]
                y_smooth = gaussian_filter1d(y_vals, sigma=2)
                axs[i].plot(x_vals, y_smooth, label=model, color=data['color'], linestyle=data['line_style'], linewidth=1.5)
        
        # Customize the plots
        for i, label in enumerate(labels):
            # axs[i].set_xlabel(('Passage' if length_of == 'source' else 'Summary') + ' Length (# of words)')
            if mode == 'ratio':
                # axs[i].set_ylabel(f'Ratio (%) of {label}')
                # axs[i].set_title(f'{label} Ratio vs Source Length')
                axs[i].set_title(f'{label}')
            elif mode == 'count':
                axs[i].set_ylabel(f'{label} Count')
                axs[i].yaxis.get_major_locator().set_params(integer=True)
                # axs[i].set_title(f'{label} Count vs Source Length')
            else:
                    raise Exception("Only \'ratio\' and \'count\' modes are supported")
            
            axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add shared legend at the bottom of the figure
        handles, labels = axs[0].get_legend_handles_labels()
        unique_handles_labels = dict(zip(labels, handles))
        fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='lower center', fontsize=9, title_fontsize='small', ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.1))
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust layout to fit the legend
        plt.show()
