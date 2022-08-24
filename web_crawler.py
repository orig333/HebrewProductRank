import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pickle


def create_soup_from_url(url):
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}
    req = requests.get(url, headers=headers).text
    return BeautifulSoup(req, 'html.parser'), req


def crawl_products_image(df):
    df = df.reindex(columns=df.columns.tolist() + ['img_url'])
    for index, row in tqdm(df.iterrows()):
        soup, req = create_soup_from_url('https://www.zap.co.il/'+ row['url'])
        productPic = soup.find_all('div', attrs={'class': 'ProductPic'})[0]
        df['img_url'][index]= productPic.img.attrs['src']
        time.sleep(1)  # keep one second between queries so the website will not be overloaded
    df.to_csv("csvs\\products_reviews_new.csv")

def get_products_review_urls(soup):
    Reviews = soup.find_all('div', attrs={'class': 'Reviews'})
    Reviews_url =[]
    for review in Reviews:
        review_a = review.a
        if review_a:
            Reviews_url.append(review_a['href'])
    return Reviews_url


def crawl_product_review_urls(product_review_url_df):
    reviews_soups, reviews_reqs = [], []
    for url in tqdm(product_review_url_df['url'], desc=f'Collecting data from {len(product_review_url_df["url"])} urls'):
        url = "https://www.zap.co.il" + url
        soup, req = create_soup_from_url(url)
        reviews_soups.append(soup), reviews_reqs.append(req)
        time.sleep(1)  # keep one second between queries so the website will not be overloaded
    with open("csvs/reviews_reqs.pickle", 'wb') as handle:  # saving the reqs objects in case we would want to collect more information (soups objects can't be saved with pickle)
        pickle.dump(reviews_reqs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"dumped {len(reviews_reqs)} urls objects")

    reviews = []
    for i, soup in tqdm(enumerate(reviews_soups), desc=f'Parsing each product reviews soup object '):
        ReviewMainTxt = soup.find_all('div', attrs={'class': 'ReviewMainTxt'})
        rate = soup.find_all('div', attrs={'class': 'RateText'})
        all_opinions = soup.find_all('div', attrs={'class': 'opinions'})
        cur_url = product_review_url_df['url'][i]
        cur_category = product_review_url_df['category'][i]
        for i, a in enumerate(ReviewMainTxt):
            r = {}
            r['title'] = soup.find("h1").text
            r['category'] = cur_category
            r['ReviewMainTxt'] = a.span.get_text(separator=". ")
            r['Rate'] = int(rate[i].span.text)
            opinions = all_opinions[i].find_all('div', attrs={'class': 'opinion'})
            for opinion in opinions:  # the reviewer add a list of positive/negative/netural aspects of the product
                text = opinion.find('div', attrs={'class': 'text'}).text
                image_path = opinion.img["src"]
                if 'positive' in image_path:
                    r['positive'] = text
                elif 'negative' in image_path:
                    r['negative'] = text
                elif 'netural' in image_path:
                    r['netural'] = text
            r['url'] = cur_url
            reviews.append(r)
    return reviews


def create_reviews_df(products_reviews_urls_path, out_df_path):
    products_reviews_urls_df = pd.read_csv(products_reviews_urls_path)
    reviews_dict = crawl_product_review_urls(products_reviews_urls_df)
    print(f"collected {len(reviews_dict)} reviews")
    reviews_df = pd.DataFrame(reviews_dict)
    reviews_df.to_csv(out_df_path)


def parse_category(category_url):
    review_urls = []
    next_page = True
    next_url = category_url
    while next_page:
        next_page = False
        soup, req = create_soup_from_url(next_url)
        review_urls += get_products_review_urls(soup)
        # get next url
        page_numbers = soup.find_all('div',attrs={'class':'NumRow'})
        for div in page_numbers:
            links = div.findAll('a',{'href': True,'aria-label': True})
            for a in links:
                if a['aria-label'] == "עבור לדף הבא":
                    next_url = 'https://www.zap.co.il' + a['href']
                    next_page = True
    return review_urls


def crawl_zap_url_categories(categories, root_url):
    products_reviews_url_dict = {'url':[],'category':[]}
    for category in categories:
        category_url = root_url + category
        products_review_urls = parse_category(category_url)
        #category_reviews = crawl_product_review_page(products_review_urls)
        products_reviews_url_dict['category'] += [category]*len(products_review_urls)
        products_reviews_url_dict['url'] += products_review_urls
        products_reviews_urls_df = pd.DataFrame(products_reviews_url_dict)
        products_reviews_urls_df.to_csv("csvs\\zap_products_reviews_urls.csv")