import json
import os
import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

def read_annotation(file_path, skip_sample_ids=[], skip_meta_sample_ids=[]):
    annotators_records = {} # annotator: {sample1: [label], sample2: [label], ...}
    data = json.load(open(file_path))
    meta_sample_ids = []
    for sample in data:
        annotations = sample['annotations']
        # skip certain samples
        sample_id = sample['sample_id'] # the id in batch
        if skip_sample_ids and sample_id in skip_sample_ids:
            continue
        meta_sample_id = sample['meta_sample_id'] # global id
        if skip_meta_sample_ids and meta_sample_id in skip_meta_sample_ids:
            continue
        meta_sample_ids.append(meta_sample_id)
        for annotation in annotations:
            annotator = annotation['annotator'] if not annotation['annotator_name'] else annotation['annotator_name'].split()[0].lower()
            if annotator not in annotators_records:
                annotators_records[annotator] = {}
            if meta_sample_id not in annotators_records[annotator]:
                annotators_records[annotator][meta_sample_id] = []
            annotators_records[annotator][meta_sample_id] += annotation['label']
        
    return meta_sample_ids, annotators_records

def compute_interannotator_agreement(
        file_path, 
        label_map, 
        level_of_measurement='interval', 
        selected_annotators = None, 
        skip_sample_ids=[], 
        skip_meta_sample_ids=[]
    ):
    sample_ids, annotators_records = read_annotation(file_path, skip_sample_ids=skip_sample_ids, skip_meta_sample_ids=skip_meta_sample_ids)
    sample_ids = sorted(sample_ids)
    print(file_path)
    
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

    # print(len(list(results.values())[0]))
    # print(results)
    print(f"{level_of_measurement} Krippendorff\'s alpha for label map\n{label_map}")
    agreement = krippendorff.alpha(np.array(list(results.values()), dtype=np.dtype(float)), level_of_measurement=level_of_measurement)
    print(round(agreement,3))

class DetectorEvaluator():
    def __init__(self, result_files, skip_sample_ids={}, skip_meta_sample_ids=[], selected_annotators={}):
        '''
        result_files: the list of files to process
        skip_sample_ids: batch sample id of samples to be skipped when computing the results
            {filename: [sample ids to skip in the file]}
        skip_meta_sample_ids: meta sample id of samples to be skipped
        selected_annotators: selected annotator for each file
            {filename: [selected annotators]}
        '''
        self.detectors = ["hhemv1", "hhem-2.1", "hhem-2.1-english", "trueteacher", "true_nli", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"]
        self.predictions = {detector: [] for detector in ['human'] + self.detectors}
        self.result_files = result_files
        self.skip_sample_ids = skip_sample_ids
        self.skip_meta_sample_ids = skip_meta_sample_ids
        self.selected_annotators = selected_annotators

    def process_results(self):
        for file_path in self.result_files:
            
            data = json.load(open(file_path))
            selected_annotators = None
            if file_path in self.selected_annotators:
                selected_annotators = self.selected_annotators[file_path]
        
            for sample in data:
                sample_id = sample['sample_id']
                if file_path in self.skip_sample_ids and sample_id in self.skip_sample_ids[file_path]:
                    continue
                meta_sample_id = sample['meta_sample_id']
                if meta_sample_id in self.skip_meta_sample_ids:
                    continue
                
                annotations = sample['annotations']
                sample_annotations = []
                for annotation in annotations:
                    if selected_annotators:
                        annotator = annotation['annotator'] if not annotation['annotator_name'] else annotation['annotator_name'].split()[0].lower()
                        if annotator in selected_annotators:
                            sample_annotations.extend(annotation['label'])
                    else:
                        sample_annotations.extend(annotation['label'])
                sample_annotations = set(sample_annotations)
                # human annotation
                if "Unwanted" in sample_annotations or 'Questionable' in sample_annotations:
                    self.predictions['human'].append(0)
                else:
                    self.predictions['human'].append(1)

                for detector in self.detectors:
                    detector_pred = sample[f"meta_{detector}"]
                    if 'hhem' in detector:
                        detector_pred = 0 if sample[f"meta_{detector}"] < 0.5 else 1
                    self.predictions[detector].append(detector_pred)
        self.pred_df = pd.DataFrame(self.predictions)
        print(self.pred_df.shape)

    def compute_correlation(self, correlation_method):
        return self.pred_df.corr(method=correlation_method)
    
    def compute_performance(self):
        self.performance_results = {}
        for detector in self.detectors:
            detector_results = {
                "ba": round(balanced_accuracy_score(self.pred_df['human'], self.pred_df[detector])*100,2),
                "f1-macro": round(f1_score(self.pred_df['human'], self.pred_df[detector], pos_label=1, average="macro")*100,2),
                "f1-halu": round(f1_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                "pr-halu": round(precision_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                're-halu': round(recall_score(self.pred_df['human'], self.pred_df[detector], pos_label=0)*100,2),
                "f1-cons": round(f1_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2),
                "pr-cons": round(precision_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2),
                're-cons': round(recall_score(self.pred_df['human'], self.pred_df[detector], pos_label=1)*100,2)
            }
            self.performance_results[detector] = detector_results
        # return self.performance_results
        return pd.DataFrame.from_dict(self.performance_results, orient='index')
    

class HaluEvaluator():
    def __init__(self, result_files, skip_sample_ids={}, skip_meta_sample_ids=[], selected_annotators={}):
        '''
        result_files: the list of files to process
        skip_sample_ids: batch sample id of samples to be skipped when computing the results
            {filename: [sample ids to skip in the file]}
        skip_meta_sample_ids: meta sample id of samples to be skipped
        selected_annotators: selected annotator for each file
            {filename: [selected annotators]}
        '''
        self.result_files = result_files
        self.skip_sample_ids = skip_sample_ids
        self.skip_meta_sample_ids = skip_meta_sample_ids
        self.selected_annotators = selected_annotators
        self.models = [
            "openai/GPT-3.5-Turbo",
            "openai/gpt-4o",
            "Qwen/Qwen2.5-7B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "cohere/command-r-08-2024",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "google/gemini-1.5-flash-001",
            "Anthropic/claude-3-5-sonnet-20240620",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        self.model_preds = {model:[] for model in self.models}

    def process_results(self):
        for file_path in self.result_files:
            
            data = json.load(open(file_path))
            selected_annotators = None
            if file_path in self.selected_annotators:
                selected_annotators = self.selected_annotators[file_path]
        
            for sample in data:
                sample_id = sample['sample_id']
                if file_path in self.skip_sample_ids and sample_id in self.skip_sample_ids[file_path]:
                    continue
                meta_sample_id = sample['meta_sample_id']
                if meta_sample_id in self.skip_meta_sample_ids:
                    continue
                
                model_name = sample['meta_model']
                annotations = sample['annotations']
                sample_annotations = []
                for annotation in annotations:
                    if selected_annotators:
                        annotator = annotation['annotator'] if not annotation['annotator_name'] else annotation['annotator_name'].split()[0].lower()
                        if annotator in selected_annotators:
                            sample_annotations.extend(annotation['label'])
                    else:
                        sample_annotations.extend(annotation['label'])
                sample_annotations = set(sample_annotations)
                # human annotation
                if "Unwanted" in sample_annotations or 'Questionable' in sample_annotations:
                    sample_pred = 0
                else:
                    sample_pred = 1
                self.model_preds[model_name].append(sample_pred)
    
    def compute_halu_rate(self):
        self.model_results = {}
        for model, preds in self.model_preds.items():
            print(f'number of records for {model}:',len(preds))
            self.model_results[model] = round((1 - sum(preds)/len(preds))*100,2)
        self.model_results = {k: v for k, v in sorted(self.model_results.items(), key=lambda item: item[1])}
        return self.model_results