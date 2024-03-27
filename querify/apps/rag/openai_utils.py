import json
from pprint import pformat

import httpx
from httpx._transports.default import ResponseStream


class _LoggingStream(ResponseStream):
    def __iter__(self):
        for chunk in super().__iter__():
            print(f"{chunk.decode()}")
            yield chunk


def no_accept_encoding(request: httpx.Request):
    request.headers.pop("accept-encoding", None)


def log_response(response):

    request = response.request
    print(f"\nRequest: {request.method} {request.url}")
    print(pformat(json.loads(request.content)), '\n')

    print('Response:', end='')
    response.stream._stream = _LoggingStream(response.stream._stream)
    print('')


openai_http_client = httpx.Client(
    event_hooks={
        "request": [no_accept_encoding],
        "response": [log_response],
    }
)