import math
import re
from collections import Counter


URL_REGEX = re.compile(r"(https?://|www\.)", re.IGNORECASE)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def contains_url(text: str) -> int:
    if not text:
        return 0
    return 1 if URL_REGEX.search(text) else 0