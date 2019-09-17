import asyncio
import re
import sys
from bs4 import BeautifulSoup

from webscraper.webscraper import HandlerDispatcher, WebScraper, \
    EVENT_BEFORE_REQUEST, EVENT_AFTER_REQUEST
from webscraper.link_spider import link_spider
from webscraper.helpers import get_node_flat_string, append_to_file, \
    url_filter
from webscraper.logger import logger, http_logger

if sys.platform not in ('win32', 'cygwin', 'cli'):
    import uvloop

    policy = uvloop.EventLoopPolicy()
    asyncio.set_event_loop_policy(policy)


def data_work_ua_jobs_extractor(url, state, filename):
    html_content = state[url]
    soup = BeautifulSoup(html_content, 'lxml')

    card = soup.find('div', {'class': 'card wordwrap'})

    if not card:
        return

    it_job_node = soup.find(it_job_node_filter)

    if not it_job_node:
        return

    content = get_node_flat_string(card)
    job_description = get_job_description(content)

    if job_description:
        append_to_file(filename, job_description)


def it_job_node_filter(tag):
    try:
        category_link = tag.name == 'a' and 'IT' in tag.get_text()
        wrapper_node = tag.parent.parent.parent.find('h5')
        category_header = 'Вакансии в категор' in wrapper_node.get_text()

        return category_link and category_header
    except AttributeError:
        return False


def get_job_description(content):
    start_separator = 'Описание вакансии '
    end_separator = ' Отправить резюме'

    start_index = content.find(start_separator) + len(start_separator)
    end_index = content.find(end_separator)

    if start_index < 0 or end_index < 0:
        return None

    return content[start_index:end_index]


def scrap():
    urls = ['https://www.work.ua/jobs-it/?category=1']
    dispatcher = HandlerDispatcher()
    scraper = WebScraper(urls, dispatcher)

    filter_pattern = re.compile(
        '^https:\/\/www.work.ua\/(jobs-it\/|jobs\/\d+\/$)')
    dispatcher.register(url_filter, EVENT_BEFORE_REQUEST,
                        pattern=filter_pattern)
    dispatcher.register(http_logger, EVENT_AFTER_REQUEST)
    dispatcher.register(link_spider, EVENT_AFTER_REQUEST,
                        scraper=scraper)
    dispatcher.register(data_work_ua_jobs_extractor, EVENT_AFTER_REQUEST,
                        filename='it_jobs.txt')

    logger.info('Start scrapping....')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(scraper.travel())
    loop.close()
    logger.info('Completed.')


if __name__ == '__main__':
    scrap()
