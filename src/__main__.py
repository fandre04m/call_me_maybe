#!/usr/bin/env python3
from pathlib import Path
import argparse
from src.file_handling import (
    load_func_definitions,
    load_prompts,
    write_call_result,
    CallResult
)
from pydantic import ValidationError
import json


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    args = parser.parse_args()

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
        functions = load_func_definitions(Path(args.functions_definition))
        prompts = load_prompts(Path(args.input))
        write_call_result(mock_results, Path(args.output))
    except FileNotFoundError as e:
        print(f"Error: file not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON - {e}")
    except ValidationError as e:
        print(f"Error: data not compatible - {e}")
    except (PermissionError, OSError) as e:
        print(f"Error: file system - {e}")


main()
