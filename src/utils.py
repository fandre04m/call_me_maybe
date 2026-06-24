from typing import List
import re

NOT_NEEDED = [
    "all", "with", "in", "on", "at", "the", "a", "an",
    "and", "or", "of", "to", "for", "by", "is", "are", "be",
    "every", "each", "using", "use",
    "where", "that", "this", "it", "its"
]


def filter_func_name(name: str, options: List[str]) -> List[str]:
    words = name.split("_")
    return [s for s in options if s.lower() not in words]


def extract_strings(prompt: str) -> List[str]:
    opts = []

    quoted = [m[1] for m in re.findall(r'(["\'])(.*?)\1', prompt)]
    if quoted:
        opts.extend(quoted)

    no_quotes = re.sub(r'(["\'])(.*?)\1', '', prompt)
    words = no_quotes.split()
    opts.extend(words)

    filtered = [s for s in opts if s.lower() not in NOT_NEEDED]

    return filtered
