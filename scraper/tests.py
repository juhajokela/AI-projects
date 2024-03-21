import unittest

from utils import parse_url


class TestUrlParser(unittest.TestCase):


    urls = [
        'https://docs.llamaindex.ai/en/stable/index.html',
        'http://docs.llamaindex.ai/en/stable/index.html',
        'docs.llamaindex.ai/en/stable/index.html',

        'https://docs.llamaindex.ai/en/stable/',
        'http://docs.llamaindex.ai/en/stable/',
        'docs.llamaindex.ai/en/stable/',

        'https://docs.llamaindex.ai/en/stable',
        'http://docs.llamaindex.ai/en/stable',
        'docs.llamaindex.ai/en/stable',

        'https://docs.llamaindex.ai/en/',
        'http://docs.llamaindex.ai/en/',
        'docs.llamaindex.ai/en/',

        'https://docs.llamaindex.ai/en',
        'http://docs.llamaindex.ai/en',
        'docs.llamaindex.ai/en',

        'https://docs.llamaindex.ai/',
        'http://docs.llamaindex.ai/',
        'docs.llamaindex.ai/',

        'https://docs.llamaindex.ai',
        'http://docs.llamaindex.ai',
        'docs.llamaindex.ai',
    ]

    expectected_output = [
        ('docs.llamaindex.ai', 'en/stable', 'index.html'),
        ('docs.llamaindex.ai', 'en/stable', 'index.html'),
        ('docs.llamaindex.ai', 'en/stable', 'index.html'),
        ('docs.llamaindex.ai', 'en', 'stable'),
        ('docs.llamaindex.ai', 'en', 'stable'),
        ('docs.llamaindex.ai', 'en', 'stable'),
        ('docs.llamaindex.ai', 'en', 'stable'),
        ('docs.llamaindex.ai', 'en', 'stable'),
        ('docs.llamaindex.ai', 'en', 'stable'),
        ('docs.llamaindex.ai', '', 'en'),
        ('docs.llamaindex.ai', '', 'en'),
        ('docs.llamaindex.ai', '', 'en'),
        ('docs.llamaindex.ai', '', 'en'),
        ('docs.llamaindex.ai', '', 'en'),
        ('docs.llamaindex.ai', '', 'en'),
        ('docs.llamaindex.ai', '', ''),
        ('docs.llamaindex.ai', '', ''),
        ('docs.llamaindex.ai', '', ''),
        ('docs.llamaindex.ai', '', ''),
        ('docs.llamaindex.ai', '', ''),
        ('docs.llamaindex.ai', '', ''),
    ]

    def test(self):
        for url, expected in zip(self.urls, self.expectected_output):
            result = parse_url(url)
            self.assertEqual(result, expected)
            print(url.ljust(50), '->', result)


if __name__ == '__main__':
    unittest.main()
