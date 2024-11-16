import os
import pathlib
import llm
import click
import json
import math
import time
from typing import List, Dict, Optional, Tuple
import sys

import sqlite_utils


def user_dir():
    llm_user_path = os.environ.get("LLM_USER_PATH")
    if llm_user_path:
        path = pathlib.Path(llm_user_path)
    else:
        path = pathlib.Path(click.get_app_dir("io.datasette.llm"))
    path.mkdir(exist_ok=True, parents=True)
    return path

def logs_db_path():
    return user_dir() / "logs.db"

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("content", type=str, required=False, nargs=-1)
    @click.option("-c", "--classes", required=True, multiple=True, help="Class options for classification")
    @click.option("-m", "--model", default="gpt-3.5-turbo", help="LLM model to use")
    @click.option("-t", "--temperature", type=float, default=0, help="Temperature for API call")
    @click.option(
        "-e", "--examples", 
        multiple=True, 
        type=str,
        help="Examples in the format 'content:class'. Can be specified multiple times."
    )
    @click.option("-p", "--prompt", help="Custom prompt template")
    @click.option("--no-content", is_flag=True, help="Exclude content from the output")
    def classify(content: Tuple[str], classes: Tuple[str], model: str, temperature: float, examples: Tuple[str], prompt: Optional[str], no_content: bool):
        """Classify content using LLM models"""
        if len(classes) < 2:
            raise click.ClickException("At least two classes must be provided")
        
        if temperature < 0 or temperature > 1:
            raise click.ClickException("Temperature must be between 0 and 1")

        # Handle piped input if no content arguments provided
        if not content and not sys.stdin.isatty():
            content = [line.strip() for line in sys.stdin.readlines()]
        elif not content:
            raise click.ClickException("No content provided. Either pipe content or provide it as an argument.")

        examples_list = None
        if examples:
            examples_list = []
            for example in examples:
                try:
                    example_content, class_ = example.rsplit(':', 1)
                    examples_list.append({"content": example_content.strip(), "class": class_.strip()})
                except ValueError:
                    click.echo(f"Warning: Skipping invalid example format: {example}", err=True)
                    continue
        
        results = classify_content(
            list(content), list(classes), model, temperature, examples_list, prompt, no_content
        )
        click.echo(json.dumps(results, indent=2))

def classify_content(
    content: List[str],
    classes: List[str],
    model: str,
    temperature: float,
    examples: Optional[List[Dict[str, str]]] = None,
    custom_prompt: Optional[str] = None,
    no_content: bool = False
) -> List[Dict[str, Optional[str]]]:
    results = []
    for item in content:
        winner, probability = get_class_probability(
            item, classes, model, temperature, examples, custom_prompt
        )
        result = {"class": winner, "score": probability}
        if not no_content:
            result["content"] = item
        results.append(result)
    return results

def get_class_probability(
    content: str,
    classes: List[str],
    model: str,
    temperature: float,
    examples: Optional[List[Dict[str, str]]] = None,
    custom_prompt: Optional[str] = None
) -> Tuple[str, float]:
    llm_model = llm.get_model(model)
    
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = f"""You are a highly efficient content classification system. Your task is to classify the given content into a single, most appropriate category from a provided list.
<INSTRUCTIONS>
1. Read and understand the content thoroughly.
2. Consider each category and how well it fits the content.
3. Choose the single most appropriate category that best describes the main theme or purpose of the content.
4. If multiple categories seem applicable, select the one that is most central or relevant to the overall message.

Here are the categories you can choose from:
<CLASSES>
{'\n'.join(classes)}
</CLASSES>

</INSTRUCTIONS>
"""

    if examples:
        prompt += "Examples:"
        for example in examples:
            prompt += f"""
    Content: {example['content']}
    Class: {example['class']}"""

    prompt += f"""</INSTRUCTIONS>
Content: {content}
Class: """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm_model.prompt(prompt, temperature=temperature, logprobs=len(classes))
            
            db = sqlite_utils.Database(logs_db_path())
            response.log_to_db(db)
            
            
            generated_text = response.text().strip().lower()
            total_logprob = 0.0
            logprobs = response.response_json.get('logprobs', {})
            
            for token_info in logprobs['content']:
                total_logprob += token_info.logprob

            probability = math.exp(total_logprob)
            
            # Ensure generated text matches one of the classes
            found_class = None
            for class_ in classes:
                if class_.lower() == generated_text:
                    found_class = generated_text
                    break
            
            if found_class is None:
                return generated_text, 0.0

            return found_class, probability

        except Exception as e:
            if attempt < max_retries - 1:
                click.echo(f"An error occurred: {e}. Retrying...", err=True)
                time.sleep(2 ** attempt)
            else:
                click.echo(f"Max retries reached. An error occurred: {e}", err=True)
                return "Error", 0

@llm.hookimpl
def register_models(register):
    pass  # No custom models to register for this plugin

@llm.hookimpl
def register_prompts(register):
    pass  # No custom prompts to register for this plugin