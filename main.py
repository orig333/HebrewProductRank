import stanza
from web_crawler import crawl_zap_url_categories, create_reviews_df,crawl_products_image
import pandas as pd
import numpy as np
from Evaluator import Evaluator
from Models import Models,AlephBERTModel
from preprocess import text_lemmatization, get_product_features_yap,get_product_features_stanza, Processor, compute_products_features_and_scores
from utils import *
from sklearn.model_selection import train_test_split
from preprocess import balance_train_set
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def crawl_data():
    zap_categories = ['c-pclaptop', 'e-headphone', 'e-cellphone', 'e-shavingmachine', 'e-coffeemachine',
                      'e-vaccumcleaner', 'e-fridge', 'e-mixer','e-camera','e-hobs','e-tv','e-iron','e-microwaveoven']
    crawl_zap_url_categories(zap_categories, root_url='https://www.zap.co.il/models.aspx?sog=')
    create_reviews_df("csvs\\zap_products_reviews_urls.csv","csvs\\products_reviews.csv")

if __name__ == '__main__':
    #crawl_data()
    df = pd.read_csv("csvs/products_reviews.csv")
    df["label"] = df['Rate'].apply(quantize_rank)
    X, y = df[["ReviewMainTxt", "url"]], df.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, y_train = balance_train_set(X_train,y_train,verbose=False)

    models = Models()
    models.train_models(X_train, y_train)

    evaluator = Evaluator(X_test, y_test)
    evaluator.evaluate(models.get_models())










