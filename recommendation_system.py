import json
import streamlit as st
import pandas as pd
import numpy as np
import re


def get_recommendation_system_stopwords():
    stop_path="stopwords\\recommendation_system_stopwords.txt"
    with open(stop_path, encoding="utf-8") as in_file:
        lines=in_file.readlines()
        res=[l.strip() for l in lines]
    return res


def find_category_features(df, urls_features_dict, category, verbose=True):
    category_urls = list(set(df[df['category'] == category]['url']))
    stop_words_list = get_recommendation_system_stopwords()

    category_features_dict = {}
    for url in category_urls:
        for feature in urls_features_dict[url]:
            if feature not in stop_words_list:
                # update feature count
                if feature in category_features_dict.keys():
                    category_features_dict[feature] += urls_features_dict[url][feature]['total']
                else:
                    category_features_dict[feature] = urls_features_dict[url][feature]['total']

    filtered_features = list(dict(sorted(category_features_dict.items(), key=lambda item: item[1], reverse=True)).keys())[:10]
    if verbose:
        print(filtered_features)

    return filtered_features


def sort_urls_by_features(df, urls_features_dict, required_features, category, verbose):
    category_urls = dict.fromkeys(list(set(df[df['category'] == category]['url'])))

    for i, url in enumerate(category_urls.keys()):
        category_urls[url] = 0
        for req_feature in required_features:
            if req_feature in urls_features_dict[url].keys():
                if category_urls[url] == 0:
                    category_urls[url] = urls_features_dict[url][req_feature]['score']
                else:
                    category_urls[url] *= urls_features_dict[url][req_feature]['score']
            else:
                # we have no information our best guess is 1/2
                if category_urls[url] == 0:
                    category_urls[url] = 0.5
                else:
                    category_urls[url] *= 0.5


    sorted_urls = dict(sorted(category_urls.items(), key=lambda item: item[1], reverse=True))
    if verbose:
        print(sorted_urls)

    return sorted_urls


def main_recommendation_function(df):
    with open('urls_features_dict.json') as json_file:
        urls_features_dict = json.load(json_file)

    st.title("Revie - You know what I like:sunglasses:")

    # get category from user
    categories = {'Cellphones':'e-cellphone',
                  'Headphones':'e-headphone',
                  'Vacuum Cleaners':'e-vaccumcleaner',
                  'Mixers':'e-mixer',
                  'Coffee Machines':'e-coffeemachine',
                  'Refrigerators':'e-fridge',
                  'Microwave Ovens':'e-microwaveoven',
                  'Shaving Machine':'e-shavingmachine',
                  'Cameras':'e-camera',
                  'TVs':'e-tv',
                  'Hobs':'e-hobs',
                  'Irons':'e-iron',
                  'Laptops':'c-pclaptop'}
    key = st.sidebar.selectbox('Choose category:',categories.keys())
    category = categories[key]

    # get category features list and let user choose important features from it
    if category:
        filtered_features = find_category_features(df, urls_features_dict, category, verbose=False)
        user_features = st.sidebar.multiselect("What features are important for you in the product?",filtered_features)

    # sort category urls by user features and display the result
    if category and user_features:
        sorted_urls = sort_urls_by_features(df,urls_features_dict,user_features,category,verbose=False)
        col1, col2, col3 = st.columns([0.35,5,1.3])
        col1.subheader(":link:")
        col2.subheader("Links")
        col3.subheader('% Match')
        for url in sorted_urls.keys():
            full_url = 'https://www.zap.co.il/' + url
            product_name = re.sub('[\n]', '',list(set(df[df['url'] == url]['title']))[0])
            link = f'[{product_name}]({full_url})'
            col1.image(list(set(df[df['url'] == url]['img_url']))[0],use_column_width='auto')
            col2.markdown(f"_{link}_")
            col3.markdown("**_{:.2f}%_**".format(sorted_urls[url]*100))


if __name__ == '__main__':
    np.random.seed(42)
    df = pd.read_csv("csvs\\recommendation_system_df.csv")
    main_recommendation_function(df)




