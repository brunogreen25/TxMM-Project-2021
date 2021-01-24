from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from os.path import basename
import re

csv_unprocessed_links = 'IEEE_Dataset/links.txt'
html_link = 'IEEE_Dataset/ieee_website.html'

processed_lines = []
with open(csv_unprocessed_links, 'r') as fp:
    for line in fp:
        processed_lines.append(line)

links = []
with open(html_link, 'r') as fp:
    for line in fp:
        if line not in processed_lines:
            continue

        link = re.findall('https://[^"]+', line)
        link = re.sub(r"&amp", r"&", link[0])
        links.append(link)


a1 = 'https://ieee-dataport.s3.amazonaws.com/open/14206/corona_tweets_02.csv?response-content-disposition=attachment%3B%20filename%3D%22corona_tweets_02.csv%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20210119%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210119T034426Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=f66ff808a8b7dd878cdbbf98db3452ef05f6999ccd959047618dc3dc375038f3'
a2 = 'https://ieee-dataport.s3.amazonaws.com/open/14206/corona_tweets_01.csv?response-content-disposition=attachment%3B%20filename%3D%22corona_tweets_01.csv%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20210119%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210119T042312Z/X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=b3f1e350f4eae934eede3e63d7dd46832c00be785abee2e021b2e09abb7647c7&;ya3a6qF3qAy5khIGYyeuUQOSwGcBHC0j'

csv_url = links[0]
print(csv_url)
req = requests.get(csv_url)
url_content = req.content
csv_file = open('downloaded.csv', 'wb')

csv_file.write(url_content)
csv_file.close()
