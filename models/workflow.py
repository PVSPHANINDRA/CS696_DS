import os
import json

from datetime import datetime
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import TrainingArguments
from transformers import pipeline

import pandas as pd
import torch

from prompt import get_more_data
from main import finetune, eval, calc_entropy_loss, plot as plot_data_map

from SNIPS import load_data as load_snips
from CSAbstruct import load_data as load_CSAbstruct

from sys import argv as args
from pathlib import Path

# variables
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device used:", device)


def workflow(config):

    # logging
    print("Worflow configuration:")
    print(config)

    # deconstruct config object
    pretrained_model_name_or_path: str = config['model_name_or_path']
    training_args = config['training_args']
    
    ## load dataset
    dataset = None
    if config['dataset_name'] == 'clinc_oos':
        dataset = load_dataset('clinc_oos', config['dataset_subset'])
        dataset = dataset.rename_column("intent", "label")
    elif config['dataset_name'] == 'snips':
        dataset = load_snips()
    elif config['dataset_name'] == 'CSAbstruct':
        dataset = load_CSAbstruct()
    
    if not dataset:
        raise Exception("Datasets are improper. Please provide valid ones")
        
    train_data, eval_data, test_data = dataset['train'], dataset['validation'], dataset['test']

    output_dir: str = config['workflow_output_dir']
    steps: int = config['steps']
    
    # variables
    dataset_types = ['train', 'validation', 'test']

    workflow_folder_name = "workflow" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    workflow_dir = os.path.join(output_dir, workflow_folder_name)

    # create workflow folder
    os.makedirs(workflow_dir)
        
    # create metrics folder
    metrics_columns = ['step', 'accuracy', 'macro_f1_score', 'weighted_f1_score']
    metrics_dir = os.path.join(workflow_dir, 'metrics')
    os.makedirs(metrics_dir)
    for _set in dataset_types:
        metrics = pd.DataFrame(columns = metrics_columns)
        metrics_file_path = os.path.join(metrics_dir, f'{_set}.csv')
        metrics.to_csv(metrics_file_path, index= False)

    # adding the workflow_config file to the current workflow directory
    workflow_config_file_path = os.path.join(workflow_dir, 'workflow_config.json')
    with open(workflow_config_file_path, 'w') as f:
        json.dump(config, f)

    # itearate
    for curr_step in range(steps):
        print('current workflow step', curr_step)
        
        curr_step = str(curr_step)

        # create folders for models, intent_class_analysis, dynamics, sentence_entropy, data, data_maps
        model_dir = os.path.join(workflow_dir, curr_step,  'model')
        intent_analysis_dir = os.path.join(workflow_dir, curr_step, 'intent_analysis')
        dynamics_dir = os.path.join(workflow_dir, curr_step, 'dynamics')
        entropy_dir = os.path.join(workflow_dir, curr_step, 'entropy')
        dataMaps_dir = os.path.join(workflow_dir, curr_step, 'data_maps')

        os.makedirs(model_dir)
        os.makedirs(intent_analysis_dir)
        os.makedirs(dynamics_dir)
        os.makedirs(entropy_dir)
        os.makedirs(dataMaps_dir)

        # finetune
        print('finetuning')
        training_args['output_dir'] = str(model_dir)
        model_path = finetune(pretrained_model_name_or_path, TrainingArguments(**training_args), train_data, eval_data, True, config['dynamics'], dynamics_dir)

        print('plotting data Maps')
        # plot dataMaps
        for dataset_type in config['dynamics']:
            dataset_dynamics_dir = os.path.join(dynamics_dir, dataset_type)
            title = config['dataset_name']
            if "dataset_subset" not in config:
                title += f"_{config['dataset_subset']}" 
            title += f"_{dataset_type}_set"
            plot_data_map(dataset_dynamics_dir, dataMaps_dir, title)

        # eval
        print('run evaluation')
        for _set in config['eval']:            
            intent_analysis_file_path = os.path.join(intent_analysis_dir, f'{_set}.csv')
            dataset = None
            metrics_file_path = os.path.join(metrics_dir, f'{_set}.csv')
            metrics_df = pd.read_csv(metrics_file_path)
            if _set == 'train':
                dataset = train_data
            elif _set == 'validation':
                dataset = eval_data
            elif _set == 'test':
                dataset = test_data

            acc, macro_f1, weighted_f1 =  eval(dataset, model_path, intent_analysis_file_path)
            metrics_df.loc[len(metrics_df)] = [curr_step, acc, macro_f1, weighted_f1]
            metrics_df.to_csv(metrics_file_path, index=False)

        # calculate entropy
        print('calculate cross entropy')
        for _set in config['entropy']:
            entropy_file_path = os.path.join(entropy_dir, f'{_set}.csv')
            dataset = None
            metrics_df = None
            if _set == 'train':
                dataset = train_data
            elif _set == 'validation':
                dataset = eval_data
            elif _set == 'test':
                dataset = test_data

            calc_entropy_loss(dataset, model_path, entropy_file_path)

        # generate data
        print('generate data')
        data_from = config['generate_data_from']
        intent_analysis_file_path = os.path.join(intent_analysis_dir, f'{data_from}.csv')
        entropy_file_path = os.path.join(entropy_dir, f'{data_from}.csv')

        data_dict = get_more_data(1, intent_analysis_file_path, entropy_file_path)

        data_df = pd.DataFrame(columns= ['text', 'true_label'])
        for intent in data_dict:
            temp = pd.DataFrame(data_dict[intent])
            temp.columns = ['text']
            temp['true_label'] = intent
            data_df = pd.concat([data_df, temp])

        data_df.reset_index()

        # verifier
        classifier = pipeline(config['pipeline_task'], model= model_path, device=device)
        predictions = classifier(data_df['text'].tolist(), batch_size = 16)
        predicted_labels = [p['label'] for p in predictions]

        data_df['predicted_label'] = predicted_labels

        # save generated data with predicted label
        generate_data_file_path = os.path.join(workflow_dir, curr_step, 'generated_data.csv')
        data_df.to_csv(generate_data_file_path, index=False)
        print("Generated data is saved at:", Path(generate_data_file_path).absolute())


        # augment data
        print('appending generate data to train set')
        ## formatting to match the train set
        correct_df = data_df[(data_df['true_label'] == data_df['predicted_label'])]
        correct_df.drop('predicted_label', axis=1)
        correct_df = correct_df.rename(columns={'true_label': 'label'})
        correct_df['label'] = correct_df['label'].apply(lambda label : classifier.model.config.label2id[label])

        ## create dataset for df
        aug_dataset = Dataset.from_pandas(correct_df)
        aug_dataset.features["label"] = train_data.features["label"]

        train_data = concatenate_datasets([train_data, aug_dataset])
        pretrained_model_name_or_path = model_path

if __name__ == "__main__":
    if len(args) < 2 or not os.path.exists(args[1]):
        raise Exception("Please provide config json to run config.json")
    
    config_file = args[1]
    config = None
    # Load JSON from file
    with open(config_file, 'r') as f:
      config = json.load(f)

    print('in the main')
    workflow(config)