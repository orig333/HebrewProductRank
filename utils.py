import pandas as pd
import matplotlib.pyplot as plt
import json
quantize = False
plot = False
overSample = False
do_text_lemmatization = True

if quantize:
    classes = {"positive": 2, "negative": 0, "neutral": 1}
else:
    classes = {"positive": 4, "negative": 0, "neutral": 2}
# classes = {"positive": 1, "negative": -1, "neutral": 0}
inv_classes = {}
for key in classes:
    inv_classes[classes[key]] = key


def quantize_rank(y):
    if quantize:
        if y <= 2.0:
            return classes["negative"]
        elif y >= 4.0:
            return classes["positive"]
        else:
            return classes["neutral"]
    else:
        return int(y) - 1


def create_data_for_alephbert(X_train, X_test, y_train, y_test, labels=5):
    train = pd.concat([X_train["ReviewMainTxt"], y_train], axis=1, ignore_index=True)
    train.columns = ['comment','label']
    train.reset_index(drop=True, inplace=True)
    train.to_csv(f'csvs\\train_{labels}_labels.tsv', sep = '\t',index=False)

    test = pd.concat([X_test["ReviewMainTxt"], y_test], axis=1, ignore_index=True)
    test.columns = ['comment','label']
    test.reset_index(drop=True, inplace=True)
    test.to_csv(f'csvs\\test_{labels}_labels.tsv', sep = '\t', index=False)
    test[:int(test.size/2)].to_csv(f'csvs\\dev_{labels}_labels.tsv', sep = '\t',index=False)


def plot_AlephBERT_training_loss(json_path,labels):
    with open(json_path) as f:
        data = json.load(f)
        epochs = [epoch['epoch'] for epoch in data['log_history']]
        losses = [epoch['loss'] for epoch in data['log_history']]
        plt.title(f'AlephBERT Training loss for {labels} classes')
        plt.xlabel('Epochs')
        plt.ylabel('CrossEntropy Loss')
        plt.plot(epochs, losses)
        plt.savefig(f'plots\\{labels} label\\AlephBERT_training_loss_{labels}_classes.png')
        plt.show()




