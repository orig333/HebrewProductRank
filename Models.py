import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer,TrainingArguments
from transformers import DataCollatorWithPadding
from utils import *
from imblearn.over_sampling import SMOTE
import torch
from pathlib import Path
import numpy as np

import mord as m


class AlephBERTModel:
    def __init__(self,checkpoint_folder, labels=5):
        self.labels = labels
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_folder, num_labels=self.labels)
        self.tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')

    class HebrewSentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    def get_dataset(self,root_folder):
        token_root = Path(root_folder)
        train = pd.read_csv(token_root / f"train_{self.labels}_labels.tsv", sep='\t')
        dev = pd.read_csv(token_root / f"dev_{self.labels}_labels.tsv", sep='\t')
        test = pd.read_csv(token_root / f"test_{self.labels}_labels.tsv", sep='\t')

        train_encodings = self.tokenizer(train["comment"].to_list(), truncation=True)
        dev_encodings = self.tokenizer(dev["comment"].to_list(), truncation=True)
        test_encodings = self.tokenizer(test["comment"].to_list(), truncation=True)
        train_labels = train["label"].to_list()
        dev_labels = dev["label"].to_list()
        test_labels = test["label"].to_list()

        self.train_dataset = self.HebrewSentimentDataset(train_encodings, train_labels)
        self.dev_dataset = self.HebrewSentimentDataset(dev_encodings, dev_labels)
        self.test_dataset = self.HebrewSentimentDataset(test_encodings, test_labels)

        self.training_args = TrainingArguments(
            output_dir=f'./alephbert_sentiment/{self.labels}_labels_results',  # output directory
            num_train_epochs=3,  # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir=f'./alephbert_sentiment/{self.labels}_labels_logs',  # directory for storing logs
            logging_steps=10
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=self.training_args,  # training arguments, defined above
            train_dataset=self.train_dataset,  # training dataset
            eval_dataset=self.dev_dataset,  # evaluation dataset
            data_collator=self.data_collator
        )

    def predict(self):
        raw_pred, _, _ = self.trainer.predict(self.test_dataset)
        y_pred = np.argmax(raw_pred, axis=1)
        count_equals = 0
        for a, b in zip(self.test_dataset.labels, y_pred):
            if a == b:
                count_equals += 1
        print(f"{self.labels} labels accuracy={count_equals / len(y_pred)}")
        return y_pred




class Models:
    def train_models(self, X_train, y_train):
        self.count_vectorizer = CountVectorizer(ngram_range=(1,2))
        self.transformer = TfidfTransformer()
        self.models = {"LogisticRegression": LogisticRegression(), "Naive Base": MultinomialNB(), "RandomForest": RandomForestClassifier(),
                       "LogisticIT (Threshold-based)": m.LogisticIT(), "OrdinalRidge (Regression-based)": m.OrdinalRidge(), "LAD (Regression-based)": m.LAD()}
        self.pipes = {}

        t = [('vectorizer', self.count_vectorizer), ('transformer', self.transformer),
                                     #('feature_select', SelectKBest(chi2, k=20000)),
                                          ]
        if overSample:
            t.append(('smote', SMOTE()))
        for model in self.models:
            self.pipes[model] = Pipeline(t + [('algo', self.models[model])])
            self.pipes[model].fit(X_train.ReviewMainTxt, y_train)

    def predict(self, X, col_name):
        predicted = {}
        for model in self.models:
            predicted[model] = self.pipes[model].predict(X[col_name])
        return predicted

    def get_strongest_words(self, clf, label, inverse_dict, algo):
        label_name = label
        label = classes[label]
        cur_coef = clf.coef_[label]
        word_df = pd.DataFrame({"val": cur_coef}).reset_index().sort_values(["val"], ascending=[False])
        word_df.loc[:, "word"] = word_df["index"].apply(lambda v: inverse_dict[v])
        word_df.head(12)[["word", "val"]].to_csv(f"csvs/{algo}_{label_name}_strong words.csv", index=False)

    def print_strongest_words(self, label):
        inverse_dict = {self.count_vectorizer.vocabulary_[w]: w for w in self.count_vectorizer.vocabulary_.keys()}
        for algo in ["LogisticRegression", "Naive Base"]:
            print(algo, f"strongest words for {label} label ----------------------------------")
            self.get_strongest_words(self.pipes[algo]["algo"], label, inverse_dict, algo)

    def get_models(self):
        return self.pipes

