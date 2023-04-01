from bs4 import BeautifulSoup
import os
import requests

URL = "https://www.ffothello.org/"

def download_file(tgt):
    content = requests.get(f'{URL}/{tgt}')
    dst = f'data/{os.path.basename(tgt)}'
    print(f'{tgt} -> {dst}')
    with open(dst, 'wb') as f:
        f.write(content.content)

if __name__ == '__main__':
    body = requests.get(f'{URL}/informatique/la-base-wthor/')
    soup = BeautifulSoup(body.content, 'html.parser')
    for link in soup.find_all('a'):
        tgt = link.get('href')
        if tgt is not None and 'ZIP' in tgt:
            download_file(tgt)

