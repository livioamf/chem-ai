import os


def test_proxy_setup():
    use_proxy = os.getenv('USE_PROXY', 'False').lower() == 'true'
    if use_proxy:
        assert 'HTTP_PROXY' in os.environ
        assert 'HTTPS_PROXY' in os.environ
        assert 'NO_PROXY' in os.environ
    else:
        assert 'HTTP_PROXY' not in os.environ or not os.environ['HTTP_PROXY']
