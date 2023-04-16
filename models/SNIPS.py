from urllib.request import urlretrieve
from pathlib import Path
import pandas as pd

from datasets import load_dataset, DatasetDict
from transformers import TrainingArguments
from main import finetune, eval, preprocess_function, calc_entropy_loss
from sys import argv as args
import os

from datasets import DatasetDict, ClassLabel


# variables
dataset_name = 'SNIPS'
function_names = ['eval', 'finetune', 'download', 'calc_entropy_loss']
dataset_types = ["train", "validation", "test"]
snips_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/snips/'))

def download_data():
    """Downloading the snips dataset from github
    """
    SNIPS_DATA_BASE_URL = (
        "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
        "master/data/snips/")

    for dataset_type in dataset_types:
        print(f"Downloading {dataset_type} data...")
        file_name = os.path.join(snips_data_path, dataset_type + '.csv')
        if dataset_type == 'validation': # github repo has validation data as valid file name
            dataset_type = 'valid'

        tempFile, headers = urlretrieve(
            SNIPS_DATA_BASE_URL + dataset_type + "?raw=true")
        lines = Path(tempFile).read_text("utf-8").strip().splitlines()
        df = pd.DataFrame([p for p in [parse_line(line)
                          for line in lines] if p is not None])
        df.to_csv(file_name, index=False)
        print("Saved at:", file_name)


def load_data() -> DatasetDict:
    """Loading snips dataset from corresponding csv format

    Returns:
        DatasetDict: it contains train, validation, test datasets
    """
    # dataset_dict - containing the dataset type as key and value is dataset of that type
    dataset_dict = DatasetDict()
    # itreate each dataset type
    for dataset_type in dataset_types:
        file_name = os.path.join(snips_data_path, dataset_type + '.csv') # file
        if not Path(file_name).exists():
            print(
                f'{dataset_type} data is not available. Tried to find at:', file_name)
            download_data()

        # load dataset
        ds_dict = load_dataset("csv", data_files = file_name)
        
        ds = ds_dict['train'] # train is the default value when we load the dataset from csv

        # casting label column
        ds = ds.cast_column('label', ClassLabel(names=ds.unique('label')))

        # appending to dataset_dict
        dataset_dict[dataset_type] = ds

    return dataset_dict


def parse_line(line: str) -> dict({'label': str, 'text': str}):
    """paring the line of a snips csv file and converting into a dict of label and text

    Args:
        line (str): 

    Returns:
        _type_: _description_
    """
    utterance_data, intent_label = line.split(" <=> ")
    items = utterance_data.split()
    words = [item.rsplit(":", 1)[0]for item in items]
    return {
        "label": intent_label,
        "text": " ".join(words),
    }


if __name__ == "__main__":
    """
    Processing the argument parameters. Currently it requires 
    Args:
    1. function_name: there are two functions. 
      a. train - to train the model 
      b. eval - to evaluate the model
    2. dataset_type: Options are "train", "validation", "test"
    3. checkpoints_out_dir: 
      a. for train function, it is used to save the checkpoint 
      b. for eval function, it is used to pick the model
    4. predictions_out_dir: 
      a. for train function - not required
      b. for eval function - file path is required, this is evaluation metrics for each class is saved
    """

    if len(args) < 2 or args[1] not in function_names:
        raise Exception("Please provide valid function_name argument")

    # assigning the arg values into variables
    function_name = args[1]

    if function_name == 'finetune':
        if len(args) < 3:
            raise Exception("Please provide checkpoints_out_dir argument")

        checkpoints_out_dir = args[2]

        # load dataset
        dataset = load_data()

        # log statements
        print("Finetuning model: START")
        print("dataset", dataset_name)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute(),
              '-This is where the Finetuning checkpoints will be saved')
        train_data, valid_data = dataset['train'], dataset['validation']

        # tokenizing the data
        train_data = train_data.map(preprocess_function, batched=True)
        valid_data = valid_data.map(preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir=checkpoints_out_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=5,
            per_device_eval_batch_size=5,
            num_train_epochs=10,
            weight_decay=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=50
        )

        finetune(training_args, train_data, valid_data)
        print("Finetuning model: END")

    elif function_name == 'eval':
        if len(args) < 3 or args[2] not in dataset_types:
            raise Exception("Please provide valid dataset_type argument")
        
        if len(args) < 4:
            raise Exception("Please provide checkpoints_out_dir argument")

        if len(args) < 5:
            raise Exception("Please provide predictions_out_dir argument")

        dataset_type = args[2]
        checkpoints_out_dir = args[3]
        predictions_out_dir = args[4]

        # load dataset
        dataset = load_data()

        # log statements
        print("Evaluating model: START")
        print("dataset", dataset_name)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute())

        dataset = dataset[dataset_type]
        eval(dataset, checkpoints_out_dir, predictions_out_dir)

        print("evaluating model: END")

    elif function_name == 'download':
        print("Downloading dataset: START")
        download_data()
        print("Downloading dataset: END")

    elif function_name == 'calc_entropy_loss':
        if len(args) < 3 or args[2] not in dataset_types:
            raise Exception("Please provide valid dataset_type argument")
        
        if len(args) < 4:
            raise Exception("Please provide checkpoints_out_dir argument")

        if len(args) < 5:
            raise Exception("Please provide entropy_analysis_path argument")

        dataset_type = args[2]
        checkpoints_out_dir = args[3]
        entropy_analysis_path = args[4]

        # load dataset
        dataset = load_data()

        # log statements
        print("Calculating entropy loss: START")
        print("dataset", dataset_name)
        print("checkpoints_out_dir", Path(checkpoints_out_dir).absolute())

        dataset = dataset[dataset_type]
        calc_entropy_loss(dataset, checkpoints_out_dir, entropy_analysis_path)

        print("Calculating entropy loss: END")   