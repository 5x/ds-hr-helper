import asyncio
from inspect import isawaitable

import aiohttp

from .logger import logger

EVENT_WORKER_START = 'worker_start'
EVENT_WORKER_END = 'worker_end'
EVENT_BEFORE_REQUEST = 'before_request'
EVENT_AFTER_REQUEST = 'after_request'


class ScraperDataState(object):
    def __init__(self):
        self.data = {}
        self.fetched_urls = {}

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]

        return None

    def __iter__(self):
        return self.data.__iter__()

    def prepare(self, url):
        self.fetched_urls[url] = None

    def save(self, url, data):
        self.data[url] = data


class WebScraper(object):
    MAX_ATTEMPTS_TO_RETRY_LOOP = 5
    DEFAULT_CONCURRENCY_LIMIT = 8
    TIMEOUT_LIMIT = 120
    DEFAULT_REQUEST_PER_MINUTE = 10000

    def __init__(self, urls, dispatcher=None, rpm=None, timeout=None):
        self.__urls = urls[:]
        self.__dispatcher = dispatcher if dispatcher else HandlerDispatcher()
        self.__rpm = rpm if rpm else self.DEFAULT_REQUEST_PER_MINUTE
        self.__retry_timeout = timeout if timeout else self.TIMEOUT_LIMIT
        self.__remaining_num_of_attempts = self.MAX_ATTEMPTS_TO_RETRY_LOOP
        self.__concurrency_limit = self.DEFAULT_CONCURRENCY_LIMIT

        self.__state = ScraperDataState()
        self.__delay_buffer_size = 0
        self.__is_forced_stop = False

    def fetch_urls(self, urls):
        self.__urls.extend(urls)

    async def travel(self):
        await self.__dispatcher.dispatch(EVENT_WORKER_START,
                                         state=self.__state)

        while not self.__is_forced_stop:
            urls = self.__get_url_batch()
            await asyncio.gather(*[self.__fetch(url) for url in urls],
                                 return_exceptions=False)
            await self.__check_timeout_limit()

        await self.__dispatcher.dispatch(EVENT_WORKER_END, state=self.__state)

    async def __fetch(self, url, **kwargs):
        self.__state.prepare(url)

        try:
            await self.__dispatcher.dispatch(EVENT_BEFORE_REQUEST,
                                             state=self.__state, url=url)
        except ValueError:
            return

        await self.__delay()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, **kwargs) as response:
                    response_text = await response.text()
                    self.__state.save(url, response_text)

                    await self.__dispatcher.dispatch(EVENT_AFTER_REQUEST,
                                                     state=self.__state,
                                                     url=url)
        except (aiohttp.client_exceptions.InvalidURL, ValueError):
            return

    async def __delay(self, min_delay=10):
        rps = 60 / self.__rpm
        self.__delay_buffer_size += rps

        if self.__delay_buffer_size >= min_delay:
            sleep_time, rest_delay = divmod(self.__delay_buffer_size, 1)
            self.__delay_buffer_size = rest_delay

            logger.info('Delay %s seconds for RPM limiting.', sleep_time)

            await asyncio.sleep(sleep_time)

    async def __check_timeout_limit(self):
        if self.__urls:
            self.__remaining_num_of_attempts = self.MAX_ATTEMPTS_TO_RETRY_LOOP
        else:
            self.__remaining_num_of_attempts -= 1

            if self.__remaining_num_of_attempts <= 0:
                self.__is_forced_stop = True

                return

            timeout_msg_template = 'Waiting for new urls, timeout %s seconds.'
            logger.info(timeout_msg_template, self.__retry_timeout)

            await asyncio.sleep(self.__retry_timeout)

    def __get_url_batch(self):
        urls_batch = set()

        while self.__urls and len(urls_batch) < self.__concurrency_limit:
            url = self.__urls.pop()

            if url not in self.__state.fetched_urls and url not in urls_batch:
                urls_batch.add(url)

        return urls_batch


class HandlerDispatcher(object):
    def __init__(self):
        self.__handlers = {}

    async def dispatch(self, channel, *args, return_exceptions=True, **kwargs):
        if channel not in self.__handlers:
            return

        for handler, bound_kwargs in self.__handlers[channel]:
            try:
                result = handler(*args, **kwargs, **bound_kwargs)

                if isawaitable(result):
                    await result
            except Exception as e:
                if return_exceptions:
                    raise e

    def register(self, handler, *channels, **kwargs):
        if not hasattr(handler, '__call__'):
            raise TypeError("Handler not callable.")

        for channel in channels:
            if channel not in self.__handlers:
                self.__handlers[channel] = []

            self.__handlers[channel].append((handler, kwargs))
