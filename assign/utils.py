import json
import os
import krippendorff
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import Counter
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

def read_annotation(file_path, skip_sample_ids=[], skip_meta_sample_ids=[]):
    annotators_records = {} # annotator: {sample1: [label], sample2: [label], ...}
    data = json.load(open(file_path))
    # meta_sample_ids = []
    sample_ids = []
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
        sample_ids.append(sample_id)
        for annotation in annotations:
            annotator = annotation['annotator'] if not annotation['annotator_name'] else annotation['annotator_name'].split()[0].lower()
            if annotator not in annotators_records:
                annotators_records[annotator] = {}
            # if meta_sample_id not in annotators_records[annotator]:
            #     annotators_records[annotator][meta_sample_id] = []
            # annotators_records[annotator][meta_sample_id] += annotation['label']
            if sample_id not in annotators_records[annotator]:
                annotators_records[annotator][sample_id] = []
            annotators_records[annotator][sample_id] += annotation['label']
        
    # return meta_sample_ids, annotators_records
    return sample_ids, annotators_records

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
    # print(sample_ids)
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
    agreement = krippendorff.alpha(np.array(list(results.values()), dtype=np.dtype(float)), level_of_measurement=level_of_measurement)
    print(round(agreement,3))
    return agreement

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
    def __init__(self, result_files, halu_labels=['Unwanted', 'Questionable'], skip_sample_ids={}, skip_meta_sample_ids=[], selected_annotators={}, num_annotators={}):
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
        self.model_map = {
            "openai/GPT-3.5-Turbo": "GPT-3.5-Turbo",
            "openai/gpt-4o": "GPT-4o",
            "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
            "microsoft/Phi-3-mini-4k-instruct": "Phi-3-mini-4k",
            "cohere/command-r-08-2024": "Command-R",
            "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
            "meta-llama/Meta-Llama-3.1-70B-Instruct":"Llama-3.1-70B",
            "google/gemini-1.5-flash-001": "Gemini-1.5-Flash",
            "Anthropic/claude-3-5-sonnet-20240620": "Claude-3.5-Sonnet",
            "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B"
        }
        self.halu_labels=halu_labels
        

    def process_results(self):
        self.model_sourcelen_preds = {model: {'source_len': [], 'summary_len': [], 'preds': [], 'avg_annotations': {l:0 for l in ['Unwanted','Questionable', 'Benign', 'Consistent']}} for model in self.models}
        self.batch_model_sourcelen_preds = {file_path:{model: {'avg_annotations': {l:0 for l in ['Unwanted','Questionable', 'Benign']}, 'preds':[], 'avg_source_len': 0, 'avg_summary_len': 0} for model in self.models} for file_path in self.result_files}
        # avg_annotation = num of one label / num of annotators
        # i.e., if one annotator made 4 unwanted annotations to a sample and the other annotator marked 2 benign labels, the labels for this sample are 2 unwanted and 1 benign
        for file_path in self.result_files:
            
            data = json.load(open(file_path))
            selected_annotators = None
            num_annotator = self.num_annotators[file_path]
            if file_path in self.selected_annotators:
                selected_annotators = self.selected_annotators[file_path]
        
            sample_count = 0
            for sample in data:
                sample_id = sample['sample_id']
                if file_path in self.skip_sample_ids and sample_id in self.skip_sample_ids[file_path]:
                    continue
                meta_sample_id = sample['meta_sample_id']
                if meta_sample_id in self.skip_meta_sample_ids:
                    continue
                sample_count += 1
                model_name = sample['meta_model']
                annotations = sample['annotations']
                self.model_sourcelen_preds[model_name]['source_len'].append(len(sample['source'].split()))
                self.model_sourcelen_preds[model_name]['summary_len'].append(len(sample['summary'].split()))
                self.batch_model_sourcelen_preds[file_path][model_name]['avg_source_len'] += len(sample['source'].split())
                self.batch_model_sourcelen_preds[file_path][model_name]['avg_summary_len'] += len(sample['summary'].split())
                
                
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
                        self.model_sourcelen_preds[model_name]['avg_annotations'][l] += annotation_counter[l] / num_annotator
                        self.batch_model_sourcelen_preds[file_path][model_name]['avg_annotations'][l] += annotation_counter[l] / num_annotator
                
                if len(occurred_annotators) < num_annotator:
                    self.model_sourcelen_preds[model_name]['avg_annotations']['Consistent'] += (num_annotator - len(occurred_annotators)) / num_annotator
                
                sample_annotations = set(sample_annotations)
                # human annotation
                if "Unwanted" in sample_annotations:
                    sample_pred = "Unwanted"
                elif 'Questionable' in sample_annotations:
                    sample_pred = "Questionable"
                elif "Benign" in sample_annotations:
                    sample_pred = "Benign"
                else:
                    sample_pred = "Consistent"
                self.model_sourcelen_preds[model_name]['preds'].append(sample_pred)
                self.batch_model_sourcelen_preds[file_path][model_name]['preds'].append(sample_pred)
    
            for model_name in self.models:
                self.batch_model_sourcelen_preds[file_path][model_name]['avg_source_len'] /= (sample_count//len(self.models))
                self.batch_model_sourcelen_preds[file_path][model_name]['avg_summary_len'] /= (sample_count//len(self.models))

    def compute_halu_rate(self):
        self.model_results = {}
        for model in self.model_sourcelen_preds:
            preds = self.model_sourcelen_preds[model]['preds']
            print(f'number of records for {model}:',len(preds))
            halu_count = 0
            for pred in preds:
                if pred in self.halu_labels:
                    halu_count += 1
            self.model_results[model] = round((halu_count/len(preds))*100,2)
        self.model_results = {k: v for k, v in sorted(self.model_results.items(), key=lambda item: item[1])}
        return self.model_results
    
    def get_halu_dist(self):
        self.model_halu_dist = {}
        for model in self.model_sourcelen_preds:
            self.model_halu_dist[model] = {}
            preds = self.model_sourcelen_preds[model]['avg_annotations']
            # print(preds, sum(list(preds.values())))
            for l in ['Unwanted','Questionable','Benign','Consistent']:
                self.model_halu_dist[model][l] = round(preds[l]/(sum(list(preds.values())))*100, 2)

        halu_dist_df = pd.DataFrame.from_dict(self.model_halu_dist, orient='index') 
        # print(halu_dist_df)
        # Set the model names as index for easier plotting
        halu_dist_df.index = halu_dist_df.index.map(lambda x: self.model_map[x])
        # halu_dist_df = halu_dist_df.sort_values(by='Unwanted', ascending=True)
        halu_dist_df = halu_dist_df.reindex(['GPT-4o', 'GPT-3.5-Turbo', "Llama-3.1-70B", "Claude-3.5-Sonnet", "Gemini-1.5-Flash", 'Llama-3.1-8B', "Qwen2.5-7B", "Command-R", "Mistral-7B", "Phi-3-mini-4k"])
        colors = [mcolors.to_rgba(c, alpha=0.7) for c in [plt.cm.tab10(3), plt.cm.tab10(1), plt.cm.tab10(0), plt.cm.tab10(2)]]
        
        # Plot the stacked bar chart
        ax = halu_dist_df.plot(kind='bar', stacked=True, figsize=(9, 7), color=colors)
        # Add numbers on the stacks
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='center', rotation=65, color='white', fontsize=13, padding=1)
    
    
        # Add labels and title
        # plt.xlabel('Model')
        plt.ylabel('Distribution of labels (%)', fontsize=14)
        # plt.title('Distribution of hallucination types')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=14, title_fontsize='small', frameon=True, ncol=4)
        plt.xticks(rotation=20, ha='right', fontsize=13)
        plt.tight_layout()
        plt.show()

    def halu_vs_length(self):
        # Initialize the plot
        plt.figure(figsize=(10, 6))
        
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
        for batch in self.batch_model_sourcelen_preds.keys():
            for model in self.batch_model_sourcelen_preds[batch].keys():
                family = model.split('/')[0]
                standard_model_name = self.model_map.get(model, model)
                if family not in family_models:
                    family_models[family] = []
                if standard_model_name not in family_models[family]:
                    family_models[family].append(standard_model_name)
        
        # Loop over each batch and model to extract data
        plot_data = {self.model_map[model]: {'halu_rate':[], 'source_len':[]} for model in self.models}
        for batch, models_data in self.batch_model_sourcelen_preds.items():
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
                predictions = data['preds']
                hallucinated = [1 if pred in self.halu_labels else 0 for pred in predictions]
                hallucinated = np.array(hallucinated)
                
                plot_data[standard_model_name]['color'] = color
                plot_data[standard_model_name]['line_style'] = line_style
                plot_data[standard_model_name]['halu_rate'].append(round(np.mean(hallucinated) * 100, 2))
                plot_data[standard_model_name]['source_len'].append(source_length)
                # Smooth the hallucination rate using a Gaussian filter
                # hallucination_rates_smooth = gaussian_filter1d(hallucination_rates, sigma=5)
        
        for model, data in plot_data.items():
            x_vals = np.array(data['source_len'])
            y_vals = np.array(data['halu_rate'])
            sorted_indices = np.argsort(x_vals)
            x_vals = x_vals[sorted_indices]
            y_vals = y_vals[sorted_indices]
            y_smooth = gaussian_filter1d(y_vals, sigma=2)
            plt.plot(x_vals, y_smooth, label=model, color=data['color'], linestyle=data['line_style'], linewidth=1.5)

        # Customize the plot
        plt.xlabel('Source Length (# words)')
        plt.ylabel('Hallucination Rate (%)')
        plt.title('Hallucination Rate vs Source Length')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.get_major_locator().set_params(integer=True)
        
        # Add shared legend at the bottom of the figure
        plt.legend(loc='lower right', fontsize='small', title_fontsize='small', ncol=4, frameon=True)
        
        plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to fit the legend
        plt.show()

    def label_vs_length(self, mode):
        # Define labels to be plotted
        labels = ['Unwanted', 'Questionable', 'Benign']
        
        # Initialize subplots for each label
        fig, axs = plt.subplots(3, 1, figsize=(6, 18))
        
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
        for batch in self.batch_model_sourcelen_preds.keys():
            for model in self.batch_model_sourcelen_preds[batch].keys():
                standard_model_name = self.model_map.get(model, model)
                family = model.split('/')[0]
                if family not in family_models:
                    family_models[family] = []
                if standard_model_name not in family_models[family]:
                    family_models[family].append(standard_model_name)
        
        # Initialize data collection for plotting
        plot_data = {label: {} for label in labels}
        
        # Loop over each batch and model to extract data
        for batch, models_data in self.batch_model_sourcelen_preds.items():
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
            if mode == 'ratio':
                axs[i].set_ylabel(f'{label} Ratio (%)')
                axs[i].set_title(f'{label} Ratio vs Source Length')
            elif mode == 'count':
                axs[i].set_ylabel(f'{label} Count')
                axs[i].yaxis.get_major_locator().set_params(integer=True)
                axs[i].set_title(f'{label} Count vs Source Length')
            
            axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add shared legend at the bottom of the figure
        handles, labels = axs[0].get_legend_handles_labels()
        unique_handles_labels = dict(zip(labels, handles))
        fig.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='lower center', fontsize='small', title_fontsize='small', ncol=4, frameon=True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust layout to fit the legend
        plt.show()