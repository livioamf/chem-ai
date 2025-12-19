import os
import urllib.parse

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope='session', autouse=True)
def setup_proxy_env():
    """
    Configura automaticamente o proxy caso USE_PROXY=True no .env.
    """
    load_dotenv('.env.proxy')
    use_proxy = os.getenv('USE_PROXY', 'False').lower() == 'true'
    if not use_proxy:
        return  # não configurar nada
    url_proxy = os.getenv('URL_PROXY')
    user = os.getenv('USER')
    password = os.getenv('PASS')
    no_proxy = os.getenv('NO_PROXY')
    if not all([url_proxy, user, password]):
        raise RuntimeError(
            'USE_PROXY=True mas URL_PROXY, USER ou PASS estão ausentes no .env'
        )
    # Codifica senha
    password_encoded = urllib.parse.quote(password)
    proxy_url = f'http://{user}:{password_encoded}@{url_proxy}'
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['NO_PROXY'] = no_proxy
    print('\n[pytest] Proxy configurado.')
