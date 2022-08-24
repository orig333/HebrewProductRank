import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer
import stanza
import requests
import json
from time import sleep
from pandas.io.json import json_normalize
from tqdm import tqdm
from sklearn.utils import resample
from utils import *
from sklearn.model_selection import train_test_split
from Models import Models
import json
from ast import literal_eval

# lematization of the corpus
class Processor:
    def __init__(self):
        self.heb_nlp = stanza.Pipeline(lang='he', processors='tokenize,mwt,pos,lemma,depparse')
        #replace MY_TOKEN with the token you got from the langndata website
        self.yap_token="e38e188b2389e98f522cea26fcf0cab0"

    def print_stanza_analysis(self, text):
        text += " XX"
        doc=self.heb_nlp(text)
        lst=[]
        for sen in doc.sentences:
            for token in sen.tokens:
                for word in token.words:
                    features=[(word.text,
                               word.lemma,
                               word.upos,
                               word.xpos,
                               word.head,
                               word.deprel,
                               word.feats)]

                    df=pd.DataFrame(features, columns=["text", "lemma", "upos", "xpos", "head", "deprel","feats"])
                    lst.append(df)
        tot_df=pd.concat(lst, ignore_index=True)
        tot_df=tot_df.shift(1).iloc[1:]
        tot_df["head"]=tot_df["head"].astype(int)
        #print(tot_df.head(50))
        return tot_df

    def print_yap_analysis(self, text):
        text= text.replace(r'"', r'\"')
        url = f'https://www.langndata.com/api/heb_parser?token={self.yap_token}'
        _json='{"data":"'+text.strip()+'"}'
#         print(url)
#         print(_json)
        headers = {'content-type': 'application/json'}
        sleep(0.5)
        r = requests.post(url,  data=_json.encode('utf-8'), headers={'Content-type': 'application/json; charset=utf-8'})
        #print(r.content)
        json_obj=r.json()
        if "md_lattice" in json_obj.keys():
            md_lattice=json_obj["md_lattice"]
            res_df=pd.io.json.json_normalize([md_lattice[i] for i in md_lattice.keys()])
            #print(res_df)
            return res_df
        elif 'msg' in json_obj.keys():
            print(f"Error:{json_obj['msg']}")
            return None
        elif 'message' in json_obj.keys():
            print(f"Error:{json_obj['message']}")
            return None



def text_lemmatization(df,out_path):
    processor = Processor()
    lemmatization_dict = {'stanza_lemmatization':[],'yap_lemmatization': []}
    for i,text in tqdm(enumerate(df['ReviewMainTxt']),desc=f'perform lemmatization of {len(df.ReviewMainTxt)} texts:'):
        lemmatization_dict['stanza_lemmatization'].append(list(processor.print_stanza_analysis(text)['lemma']))
        try:
            sleep(3)
            max_words_yap = 250
            cropped_text = " ".join(text.split()[:max_words_yap])
            lemmatization_dict['yap_lemmatization'].append(list(processor.print_yap_analysis(cropped_text)['lemma']))
        except Exception as e:
            print(f"The exception {e.__class__} occurred.")
            print("Continue to next row.")
            lemmatization_dict['yap_lemmatization'].append([])
    df['stanza_lemmatization'] = pd.Series(lemmatization_dict['stanza_lemmatization'])
    df['yap_lemmatization'] = pd.Series(lemmatization_dict['yap_lemmatization'])
    df.to_csv(out_path)


def get_product_features_stanza(df, col_name, processor, out_file_path):
    stanza_features_dict = {'features':[]}
    for index, row in df.iterrows():
        print(f'index={index}')
        if pd.isnull(row[col_name]) or len(row[col_name]) <= 1:
            stanza_features_dict['features'].append([])
            continue
        stanza_analysis = processor.print_stanza_analysis(row[col_name])
        cur_features_list = []
        if col_name == 'ReviewMainTxt':
            stanza_adj = stanza_analysis[stanza_analysis['upos'] == 'ADJ']
            for adj_index, adj_row in stanza_adj.iterrows():
                if stanza_analysis.iloc[adj_row['head'] - 1]['upos'] == 'NOUN':
                    cur_features_list.append((stanza_analysis.iloc[adj_row['head'] - 1]['lemma'], adj_row['lemma']))
        else:
            cur_features_list = list(stanza_analysis[stanza_analysis['upos'] == 'NOUN']['lemma'])
        print(cur_features_list)
        stanza_features_dict['features'].append(cur_features_list)

    stanza_features = pd.DataFrame(stanza_features_dict)
    stanza_features.to_csv(out_file_path, index=False)


def get_product_features_yap(df, col_name, processor, out_file_path):
    yap_features_dict = {'features':[]}
    max_words_yap = 250
    for index, row in df.iterrows():
        if index % 500 == 0:
            sleep(20)
        print(f'index={index}')
        if pd.isnull(row[col_name]) or len(row[col_name]) <= 1:
            yap_features_dict['features'].append([])
            continue
        sleep(3)
        cropped_text = " ".join(row[col_name].split()[:max_words_yap])
        try:
            yap_analysis = processor.print_yap_analysis(cropped_text)
            cur_features_list = []
            if col_name == 'ReviewMainTxt':
                yap_analysis_adj = yap_analysis[yap_analysis['pos'] == 'JJ']
                for adj_index, adj_row in yap_analysis_adj.iterrows():
                    if yap_analysis['pos'][int(adj_row['num_last']) - 1] == 'NN':
                        cur_features_list.append((yap_analysis['lemma'][int(adj_row['num_last']) - 1], adj_row['lemma']))
            else:
                cur_features_list = list(yap_analysis[(yap_analysis['pos'] == 'NN') | (yap_analysis['pos'] == 'NNT')]['lemma'])

            print(cur_features_list)
            yap_features_dict['features'].append(cur_features_list)

        except Exception as e:
            print(f"The exception {e.__class__} occurred.")
            print("Continue to next row.")
            yap_features_dict['features'].append([])
            continue

    yap_features = pd.DataFrame(yap_features_dict)
    yap_features.to_csv(out_file_path, index=False)


def compute_products_features_and_scores(df):
    stanza_positive_product_features = pd.read_csv("csvs\\stanza_positive_product_features.csv")
    stanza_netural_product_features = pd.read_csv("csvs\\stanza_netural_product_features.csv")
    stanza_negative_product_features = pd.read_csv("csvs\\stanza_negative_product_features.csv")
    stanza_product_features = pd.read_csv("csvs\\stanza_product_features.csv")
    yap_positive_product_features = pd.read_csv("csvs\\yap_positive_product_features.csv")
    yap_netural_product_features = pd.read_csv("csvs\\yap_netural_product_features.csv")
    yap_negative_product_features = pd.read_csv("csvs\\yap_negative_product_features.csv")
    yap_product_features = pd.read_csv("csvs\\yap_product_features.csv")

    urls_dict = dict.fromkeys(set(df['url']))
    df["label"] = df['Rate'].apply(quantize_rank)
    X, y = df[["ReviewMainTxt", "url"]], df.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    models = Models()
    models.train_models(X_train, y_train)

    for cur_url in tqdm(urls_dict.keys()):
        cur_dict = {}
        stanza_main_text_features = stanza_product_features.iloc[df[df['url'] == cur_url].index]
        stanza_positive_features = stanza_positive_product_features.iloc[df[df['url'] == cur_url].index]
        stanza_netural_features = stanza_netural_product_features.iloc[df[df['url'] == cur_url].index]
        stanza_negative_features = stanza_negative_product_features.iloc[df[df['url'] == cur_url].index]
        yap_main_text_features = yap_product_features.iloc[df[df['url'] == cur_url].index]
        yap_positive_features = yap_positive_product_features.iloc[df[df['url'] == cur_url].index]
        yap_netural_features = yap_netural_product_features.iloc[df[df['url'] == cur_url].index]
        yap_negative_features = yap_negative_product_features.iloc[df[df['url'] == cur_url].index]
        for index, row in stanza_positive_features.iterrows():
            for f in literal_eval(row['features']):
                if f in cur_dict.keys():
                    cur_dict[f]['positive'] += 1
                    cur_dict[f]['total'] += 1
                else:
                    cur_dict[f] = {}
                    cur_dict[f]['positive'] = 1
                    cur_dict[f]['netural'] = 0
                    cur_dict[f]['negative'] = 0
                    cur_dict[f]['total'] = 1

        for index, row in yap_positive_features.iterrows():
            for f in literal_eval(row['features']):
                if f in cur_dict.keys():
                    cur_dict[f]['positive'] += 1
                    cur_dict[f]['total'] += 1
                else:
                    cur_dict[f] = {}
                    cur_dict[f]['positive'] = 1
                    cur_dict[f]['netural'] = 0
                    cur_dict[f]['negative'] = 0
                    cur_dict[f]['total'] = 1

        for index, row in stanza_netural_features.iterrows():
            for f in literal_eval(row['features']):
                if f in cur_dict.keys():
                    cur_dict[f]['netural'] += 1
                    cur_dict[f]['total'] += 1
                else:
                    cur_dict[f] = {}
                    cur_dict[f]['positive'] = 0
                    cur_dict[f]['netural'] = 1
                    cur_dict[f]['negative'] = 0
                    cur_dict[f]['total'] = 1

        for index, row in yap_netural_features.iterrows():
            for f in literal_eval(row['features']):
                if f in cur_dict.keys():
                    cur_dict[f]['netural'] += 1
                    cur_dict[f]['total'] += 1
                else:
                    cur_dict[f] = {}
                    cur_dict[f]['positive'] = 0
                    cur_dict[f]['netural'] = 1
                    cur_dict[f]['negative'] = 0
                    cur_dict[f]['total'] = 1

        for index, row in stanza_negative_features.iterrows():
            for f in literal_eval(row['features']):
                if f in cur_dict.keys():
                    cur_dict[f]['negative'] += 1
                    cur_dict[f]['total'] += 1
                else:
                    cur_dict[f] = {}
                    cur_dict[f]['positive'] = 0
                    cur_dict[f]['netural'] = 0
                    cur_dict[f]['negative'] = 1
                    cur_dict[f]['total'] = 1

        for index, row in yap_negative_features.iterrows():
            for f in literal_eval(row['features']):
                if f in cur_dict.keys():
                    cur_dict[f]['negative'] += 1
                    cur_dict[f]['total'] += 1
                else:
                    cur_dict[f] = {}
                    cur_dict[f]['positive'] = 0
                    cur_dict[f]['netural'] = 0
                    cur_dict[f]['negative'] = 1
                    cur_dict[f]['total'] = 1

        for index, row in stanza_main_text_features.iterrows():
            for t in literal_eval(row['features']):
                if t[0] is None or t[1] is None:
                    continue
                input_text = t[0] + " " + t[1]
                prediction = pd.DataFrame(models.predict(pd.DataFrame({'features': [input_text]}), col_name='features')).mean(axis=1)[0]
                f = t[0]
                if f in cur_dict.keys():
                    if prediction >= 4:
                        cur_dict[f]['positive'] += 1
                        cur_dict[f]['total'] += 1
                    elif 2 < prediction < 4:
                        cur_dict[f]['netural'] += 1
                        cur_dict[f]['total'] += 1
                    elif prediction <= 2:
                        cur_dict[f]['negative'] += 1
                        cur_dict[f]['total'] += 1
                else:
                    if prediction >= 4:
                        cur_dict[f] = {}
                        cur_dict[f]['positive'] = 1
                        cur_dict[f]['netural'] = 0
                        cur_dict[f]['negative'] = 0
                        cur_dict[f]['total'] = 1
                    elif 2 < prediction < 4:
                        cur_dict[f] = {}
                        cur_dict[f]['positive'] = 0
                        cur_dict[f]['netural'] = 1
                        cur_dict[f]['negative'] = 0
                        cur_dict[f]['total'] = 1
                    elif prediction <= 2:
                        cur_dict[f] = {}
                        cur_dict[f]['positive'] = 0
                        cur_dict[f]['netural'] = 0
                        cur_dict[f]['negative'] = 1
                        cur_dict[f]['total'] = 1

        for index, row in yap_main_text_features.iterrows():
            for t in literal_eval(row['features']):
                if t[0] is None or t[1] is None:
                    continue
                input_text = t[0] + " " + t[1]
                prediction = pd.DataFrame(models.predict(pd.DataFrame({'features': [input_text]}), col_name='features')).mean(axis=1)[0]
                f = t[0]
                if f in cur_dict.keys():
                    if prediction >= 4:
                        cur_dict[f]['positive'] += 1
                        cur_dict[f]['total'] += 1
                    elif 2 < prediction < 4:
                        cur_dict[f]['netural'] += 1
                        cur_dict[f]['total'] += 1
                    elif prediction <= 2:
                        cur_dict[f]['negative'] += 1
                        cur_dict[f]['total'] += 1
                else:
                    if prediction >= 4:
                        cur_dict[f] = {}
                        cur_dict[f]['positive'] = 1
                        cur_dict[f]['netural'] = 0
                        cur_dict[f]['negative'] = 0
                        cur_dict[f]['total'] = 1
                    elif 2 < prediction < 4:
                        cur_dict[f] = {}
                        cur_dict[f]['positive'] = 0
                        cur_dict[f]['netural'] = 1
                        cur_dict[f]['negative'] = 0
                        cur_dict[f]['total'] = 1
                    elif prediction <= 2:
                        cur_dict[f] = {}
                        cur_dict[f]['positive'] = 0
                        cur_dict[f]['netural'] = 0
                        cur_dict[f]['negative'] = 1
                        cur_dict[f]['total'] = 1

        urls_dict[cur_url] = cur_dict

    # compute score
    for url in tqdm(urls_dict.keys()):
        for feature_dict in urls_dict[url].keys():
            urls_dict[url][feature_dict]['score'] = urls_dict[url][feature_dict]['positive'] / urls_dict[url][feature_dict]['total']

    with open('urls_features_dict.json', 'w') as fp:
        json.dump(urls_dict, fp)
    pd.DataFrame.from_dict(urls_dict, orient='index').to_csv('urls_features_df.csv', index=False)


def balance_train_set(X_train, y_train, majority_class=4, labels=5, verbose=True):
    if verbose:
        print('Number of samples per class in train set before oversampling:')
        for i in range(labels):
            print(f"class {i} X size: {X_train[y_train == i].shape}")
            print(f"class {i} y size: {y_train[y_train == i].shape}")

    oversampled_classes_X = []
    oversampled_classes_y = []
    for i in range(labels):
        if i != majority_class:
            cur_oversampled_X, cur_oversampled_y = resample(X_train[y_train == i], y_train[y_train == i], replace=True,
                                                            n_samples=y_train[y_train == majority_class].shape[0],
                                                            random_state=123)
            oversampled_classes_X.append(cur_oversampled_X)
            oversampled_classes_y.append(cur_oversampled_y)

    oversampled_classes_X.append(X_train[y_train == majority_class])
    oversampled_classes_y.append(y_train[y_train == majority_class])
    X_train = pd.concat(oversampled_classes_X, ignore_index=True)
    y_train = pd.concat(oversampled_classes_y, ignore_index=True)

    if verbose:
        print('Number of samples per class in train set after oversampling:')
        for i in range(labels):
            print(f"class {i} X size: {X_train[y_train == i].shape}")
            print(f"class {i} y size: {y_train[y_train == i].shape}")

    return X_train, y_train

