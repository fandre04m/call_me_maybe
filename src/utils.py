from typing import List
import re


def extract_strings(prompt: str) -> List[str] | None:
    opts = []
    quoted = [m[1] for m in re.findall(r'(["\'])(.*?)\1', prompt)]

    return opts.extend(quoted)
