from typing import List
import re

NOT_NEEDED = [
    "greet", "reverse", "the", "of", "it", "is", "square", "root",
    "in", "replace", "substitute", "calculate", "all",
    "with", "string", "word"
]


def extract_strings(prompt: str) -> List[str] | None:
    opts = []

    quoted = [m[1] for m in re.findall(r'(["\'])(.*?)\1', prompt)]
    if quoted:
        opts.extend(quoted)

    no_quotes = re.sub(r'(["\'])(.*?)\1', '', prompt)
    words = no_quotes.split()
    opts.extend(words)

    filtered = [s for s in opts if s.lower() not in NOT_NEEDED]

    return filtered
