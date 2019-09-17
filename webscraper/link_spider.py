from urllib.parse import urljoin, urldefrag, urlparse

from bs4 import BeautifulSoup


async def link_spider(url, state, scraper, host=None):
    html_content = state[url]
    soup = BeautifulSoup(html_content, 'lxml')

    links = soup.find_all('a', href=True)

    urls = []
    for page_link in links:
        href = page_link.get('href').strip()
        href = urljoin(url, href)
        href = urldefrag(href)[0]

        if host is None or urlparse(href).hostname == host:
            urls.append(href)

    scraper.fetch_urls(urls)
