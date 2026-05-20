#!/usr/bin/env python3
from loader_validator import load_func_definitions, load_prompts
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
    except (ValueError, ValidationError) as e:
        print(e)


main()
