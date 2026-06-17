#!/usr/bin/env python3
from pathlib import Path
import argparse
from src import FileLoader, GeneratorFSM
from pydantic import ValidationError
import json
from .llm_handler import LLMHandler
from src import llm_handler


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

    file_loader = FileLoader()
    try:
        file_loader.load_func_definitions(Path(args.functions_definition))
        file_loader.load_prompts(Path(args.input))
    except FileNotFoundError as e:
        print(f"Error: file not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON - {e}")
    except ValidationError as e:
        print(f"Error: data not compatible - {e}")
    except (PermissionError, OSError) as e:
        print(f"Error: file system - {e}")

    fsm = GeneratorFSM(file_loader.func_definitions)
    try:
        for prompt in file_loader.prompts:
            fsm.run(prompt.prompt)
        # fsm.run("What is my name?")
        # fsm.run(file_loader.prompts[1].prompt)
        # llm = LLMHandler(file_loader.func_definitions)
        # llm.run_prompt(file_loader.prompts[0].prompt)
            print(f"Elapsed time: {fsm.elapsed_time:.2f}")
            print()
    except ValueError as e:
        print(f"Error: {e}")


main()
