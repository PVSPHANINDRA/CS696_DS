from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from urllib.request import urlretrieve
from datasets import Dataset,DatasetDict
import pandas as pd
import sys

from transformers import AutoTokenizer,DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

from datasets import ClassLabel, Sequence
from huggingface_hub.hf_api import HfFolder

# imdb = load_dataset("imdb")

def get_dataset(dataset_name):
    if dataset_name == "SNIPS":
        SNIPS_DATA_BASE_URL = (
            "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
            "master/data/snips/"
        )
        for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
            path = Path(filename)
            if not path.exists():
                print(f"Downloading {filename}...")
                urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)

        lines_train = Path("train").read_text("utf-8").strip().splitlines()
        #lines_train[:5]

        def parse_line(line):
            utterance_data, intent_label = line.split(" <=> ")
            items = utterance_data.split()
            words = [item.rsplit(":", 1)[0]for item in items]
            word_labels = [item.rsplit(":", 1)[1]for item in items]
            return {
                "label": intent_label,
                "text": " ".join(words),
            }

        parsed = [parse_line(line) for line in lines_train]

        df_train = pd.DataFrame([p for p in parsed if p is not None])

        lines_valid = Path("valid").read_text("utf-8").strip().splitlines()
        lines_test = Path("test").read_text("utf-8").strip().splitlines()

        df_valid = pd.DataFrame([parse_line(line) for line in lines_valid])
        df_test = pd.DataFrame([parse_line(line) for line in lines_test])

        le = LabelEncoder()
        df_train.label = le.fit_transform(df_train.label)
        df_test.label = le.fit_transform(df_test.label)
        df_valid.label = le.fit_transform(df_valid.label)
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        id_mapping = {v: k for k, v in label_mapping.items()}

        dataset_train = Dataset.from_pandas(df_train)
        dataset_test = Dataset.from_pandas(df_test)
        dataset_valid = Dataset.from_pandas(df_valid)


        all_dataset = DatasetDict({"train":dataset_train, "test":dataset_test , "validation":dataset_valid})
        
    elif dataset_name == "CLINC":
        all_dataset = load_dataset("clinc_oos", "small")
        all_dataset = all_dataset.rename_column("intent", "label")
    return all_dataset

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    arg_list = sys.argv
    lr = float(sys.argv[1])
    dataset_name = sys.argv[2]
    num_epochs = int(sys.argv[3])
    all_dataset = get_dataset(dataset_name)
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    tokenized_all_dataset = all_dataset.map(preprocess_function,batched= True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    accuracy = evaluate.load("accuracy")
    
    label_names = sorted(set(labels for labels in all_dataset["train"]["label"]))
    # label_names
    # Cast to ClassLabel
    # all_dataset = all_dataset.cast_column("label", Sequence(ClassLabel(names=label_names)))


    # labels = all_dataset["train"].features["label"].names
    label2id = {"Label_"+str(i) : i for i in range(len(label_names))}
    id2label = {i: "Label_"+str(i)  for i in range(len(label_names))}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=len(label_names), id2label=id2label, label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    
    HfFolder.save_token('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')
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_all_dataset["train"],
        eval_dataset=tokenized_all_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    
    
    
    