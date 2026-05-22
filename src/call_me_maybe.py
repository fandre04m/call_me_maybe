#!/usr/bin/env python3
from file_handling import (
    load_func_definitions,
    load_prompts,
    write_call_result
)
from pydantic import ValidationError


def main() -> None:
    try:
        functions = load_func_definitions()
        prompts = load_prompts()
        for item in functions:
            print(item)
            print()
        for item in prompts:
            print(item)
            print()
        write_call_result()
    except (ValueError, ValidationError) as e:
        print(e)


main()
