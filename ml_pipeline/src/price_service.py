# price_service.py

import requests

def get_xrp_price_usd() -> float:
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ripple&vs_currencies=usd"
    data = requests.get(url).json()
    return data["ripple"]["usd"]