"""Module docstring."""

import os

MAX_RETRIES = 3


def parse_config(path):
    """Read a configuration file and return the parsed key value pairs as a dict."""
    with open(path) as fh:
        return dict(line.split("=", 1) for line in fh)


# Attached comment.
@staticmethod
def decorated(x):
    return x


class HttpClient(object):
    """A tiny HTTP client used for testing the chunker."""

    timeout = 10

    def __init__(self, base_url):
        self.base_url = base_url

    def get_json(self, route):
        """Fetch a route and decode the JSON body into Python objects for callers."""

        def helper():
            return route

        return helper()

    class Inner:
        def ping(self):
            return "pong"


if __name__ == "__main__":
    print(os.getcwd())
