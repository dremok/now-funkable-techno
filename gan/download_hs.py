import json
import os
import re
import shutil

import bs4
import requests

from definitions import ROOT_DIR

BASE_URL = 'https://api.hearthstonejson.com'
IMG_URL = 'https://wow.zamimg.com/images/hearthstone/cards/enus/original'
DATA_PATH = f'{ROOT_DIR}/gan/data/hearthstone'

res = requests.get('https://api.hearthstonejson.com/v1/')
soup = bs4.BeautifulSoup(res.text, 'html.parser')
links = [soup.select('a')]

os.makedirs(DATA_PATH, exist_ok=True)


def load_images(url):
    cards = json.loads(url.text)
    card_ids = [x['id'] for x in cards]
    for card_id in card_ids:
        img_url = f'{IMG_URL}/{card_id}.png'
        filename = f'{DATA_PATH}/{img_url.split("/")[-1]}'
        if os.path.isfile(filename):
            print(f'File already exists: {filename}')
            continue
        response = requests.get(img_url, stream=True)
        if response.status_code != 200:
            print(f"Can't download: {filename}")
            continue
        else:
            response.raw.decode_content = True
            with open(filename, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print(f'{filename} saved successfully!')


def main():
    link = '/v1/93849/enUS/'
    if re.match(r'/v1/\d+/enUS/', link):
        print(f'Downloading images from {link}')
        card_link = f'{BASE_URL}{link}cards.json'
        url = requests.get(card_link)
        load_images(url)


if __name__ == '__main__':
    main()
