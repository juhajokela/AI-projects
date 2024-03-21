import os
import sys

import scrapy
import trafilatura

from scrapy.crawler import CrawlerProcess

from utils import (
    parse_url,
    remove_protocol,
    write_file,
)


class BaseSpider(scrapy.Spider):
    name = "spider"

    def start_requests(self):
        yield scrapy.Request(url=self.start_url, callback=self.parse)

    def parse(self, response):
        domain, path, document = parse_url(response.url)

        if not document.endswith('.html'):
            document += '.html'

        if remove_protocol(response.url).startswith(remove_protocol(self.root_url)):
            write_file(os.path.join('raw_data', domain, path, document), response.text)
            # Extract the main content using Trafilatura
            main_content = trafilatura.extract(response.text, include_comments=False, include_tables=False)
            if main_content:
                # You can process the extracted content here (e.g., save to file or database)
                write_file(os.path.join('data', domain, path, document.replace('.html', '.txt')), main_content)

            # Recursively follow links
            for href in response.css('a::attr(href)').getall():
                yield response.follow(href, self.parse)


def main():
    start_url = sys.argv[1]
    root_url = start_url.rsplit('/', 1)[0] if start_url.endswith('.html') else start_url
    if input(f'start_url: {start_url}\nroot_url: {root_url}\nOk? Continue [y/n]? ').lower() != 'y':
        exit()
    Spider = type("Spider", tuple([BaseSpider]), {
        "start_url": start_url,
        "root_url": root_url,
    })
    process = CrawlerProcess()
    process.crawl(Spider)
    process.start()

if __name__ == '__main__':
    main()
