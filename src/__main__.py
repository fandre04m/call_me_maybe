#!/usr/bin/env python3
from pathlib import Path
import argparse
from src import FileHandler, PromptBuilder, ModelCaller
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

    file_handler = FileHandler()
    prompt_maker = PromptBuilder()
    model = ModelCaller()
    try:
        file_handler.load_func_definitions(Path(args.functions_definition))
        file_handler.load_prompts(Path(args.input))
        built_prompt = prompt_maker.build(
            file_handler.func_definitions,
            file_handler.prompts[0].prompt
        )
        model.run_prompt(built_prompt)
        # file_handler.write_call_result(mock_results, Path(args.output))
    except FileNotFoundError as e:
        print(f"Error: file not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON - {e}")
    except ValidationError as e:
        print(f"Error: data not compatible - {e}")
    except (PermissionError, OSError) as e:
        print(f"Error: file system - {e}")


main()
