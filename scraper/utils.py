import os
import re


def write_file(path, content):
    dir_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(path, 'w') as f:
        f.write(content)


def remove_protocol(url):
    return re.sub(r'^.*?://', '', url)


def parse_url(url):
    base_url = remove_protocol(url)
    cleaned_url = base_url[:-1] if base_url.endswith('/') else base_url
    tokens = cleaned_url.split('/')
    domain = tokens[0]

    if len(tokens) == 1:
        return domain, '', ''

    path = '/'.join(tokens[1:-1])
    document = tokens[-1]

    return domain, path, document