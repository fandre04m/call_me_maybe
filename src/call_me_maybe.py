#!/usr/bin/env python3
from file_handling import (
    load_func_definitions,
    load_prompts,
    write_call_result,
    CallResult
)
from pydantic import ValidationError


def main() -> None:
    mock_results = [
        CallResult(
            prompt="What is the sum of 2 and 3?",
            name="fn_add_numbers",
            parameters={"a": 2, "b": 3}
        ),
        CallResult(
            prompt="Reverse the string 'hello'",
            name="fn_reverse_string",
            parameters={"s": "hello"}
        )
    ]
    try:
        functions = load_func_definitions()
        prompts = load_prompts()
        for item in functions:
            print(item)
            print()
        for item in prompts:
            print(item)
            print()
        write_call_result(mock_results)
    except (ValueError, ValidationError) as e:
        print(e)


main()
