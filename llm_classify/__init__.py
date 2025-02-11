import os
import pathlib
import llm
import click
import json
import math
import time
import sys
import logging
from typing import List, Dict, Optional, Tuple

import sqlite_utils
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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

def setup_database(db_path: pathlib.Path) -> sqlite_utils.Database:
    db = sqlite_utils.Database(db_path)
    
    if "classifications" not in db.table_names():
        db["classifications"].create({
            "id": int,
            "content": str,
            "class": str,
            "model": str,
            "temperature": float,
            "timestamp": float,
            "response_json": str
        }, pk="id")
    
    return db

def process_classification_response(response) -> Tuple[Optional[str], float]:
    try:
        # Extract text and logprobs from response
        text = response.response_json['content'].strip().lower()
        logprobs_list = response.response_json['logprobs']['content']
        
        # Calculate probability from logprob
        total_logprob = sum(item.logprob for item in logprobs_list)
        probability = math.exp(total_logprob)
        
        return text, probability
    except Exception as e:
        logger.error("Error processing response: %s", e)
        return None, 0.0

def format_prompt(content: str, classes: Tuple[str], examples_list: Optional[List[Dict[str, str]]], custom_prompt: Optional[str]) -> str:
    default_prompt = """
<INSTRUCTIONS>
Your task is to classify the given content into ONE of the provided categories. Respond with ONLY the category name.

<CLASSES>
{classes}
</CLASSES>

Content: {content} 
Class:
</INSTRUCTIONS>
"""
    prompt_template = custom_prompt or default_prompt
    formatted_prompt = prompt_template.format(content=content, classes=", ".join(classes))
    if examples_list:
        formatted_prompt += "\nExamples:"
        for example in examples_list:
            formatted_prompt += f"\n  Content: {example['content']}\n  Class: {example['class']}"
    formatted_prompt += f"\nInput: {content}\nClass:"
    return formatted_prompt

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
                    logger.warning("Skipping invalid example format: %s", example)
                    continue
        
        db = setup_database(logs_db_path())
    
        results = []
        for item in content:
            try:
                formatted_prompt = format_prompt(item, classes, examples_list, prompt)

                # Use the model's prompt method with the formatted string
                response = llm.get_model(model).prompt(
                    formatted_prompt,
                    temperature=temperature,
                    top_logprobs=1,
                    max_tokens=1  # Add max_tokens limit
                )
                
                # Force evaluation of the lazy response by converting it to a string.
                evaluated_response = str(response)

                # Now process the response using the already evaluated version.
                result, probability = process_classification_response(response)
                
                if result:
                    # Convert response_json to serializable format
                    serializable_response = {}
                    if hasattr(response, 'response_json'):
                        serializable_response = _make_serializable(response.response_json)
                    
                    db["classifications"].insert({
                        "content": item if not no_content else "",
                        "class": result,
                        "model": model,
                        "temperature": temperature,
                        "timestamp": time.time(),
                        "response_json": json.dumps(serializable_response)
                    })
                    
                    result_dict = {
                        "class": result,
                        "score": probability
                    }
                    if not no_content:
                        result_dict["content"] = item
                    results.append(result_dict)
                else:
                    logger.error("Failed to get valid classification for content: %s", item)
                    
            except Exception as e:
                logger.error("Error processing item: %s", e)
                continue
        
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

def _extract_completion_logprobs(response_json: dict) -> float:
    """Extract logprobs from completion model response"""
    if not response_json.get('logprobs'):
        return 1.0
    
    total_logprob = 0.0
    for token_info in response_json['logprobs']:
        if token_info.get('top_logprobs'):
            # Get the logprob of the actual token used
            actual_token = token_info['text']
            for option in token_info['top_logprobs'][0].keys():
                if option.strip() == actual_token.strip():
                    total_logprob += token_info['top_logprobs'][0][option]
                    break
    
    return math.exp(total_logprob) if total_logprob != 0 else 1.0


def _extract_chat_logprobs(response_json: dict) -> float:
    """Extract logprobs from chat model response"""
    try:
        if 'logprobs' in response_json:
            # Handle nested content structure
            if isinstance(response_json['logprobs'], dict) and 'content' in response_json['logprobs']:
                logprobs = response_json['logprobs']['content']
            else:
                logprobs = response_json['logprobs']

            # Calculate total probability from logprobs
            total_logprob = 0.0
            for token_info in logprobs:
                if 'logprob' in token_info:
                    total_logprob += token_info['logprob']
                elif isinstance(token_info, dict) and 'top_logprobs' in token_info:
                    # Get the highest probability from top_logprobs
                    top_probs = token_info['top_logprobs']
                    if isinstance(top_probs, list) and top_probs:
                        if isinstance(top_probs[0], dict):
                            max_logprob = max(top_probs[0].values())
                        else:
                            max_logprob = max(prob.logprob for prob in top_probs)
                        total_logprob += max_logprob

            # Convert logprob to probability
            return math.exp(total_logprob)
    except Exception as e:
        logger.error("Error extracting logprobs: %s", e)
    return 1.0

def _extract_chat_content(response_json: dict) -> str:
    """Extract content from chat model response"""
    if 'content' not in response_json:
        raise ValueError("No content found in chat response")
    return response_json['content'].strip()


class ClassificationOptions(BaseModel):
    content: str
    classes: List[str]
    examples: Optional[List[Dict[str, str]]] = None

def _create_prompt(options: ClassificationOptions, custom_prompt: Optional[str] = None) -> str:
    if custom_prompt:
        return custom_prompt
        
    prompt = """You are a content classifier. Classify the content into exactly one of these classes:
{classes}

Respond with only the class name.
"""
    if options.examples:
        prompt += "\nExamples:\n"
        for ex in options.examples:
            prompt += f"Content: {ex['content']}\nClass: {ex['class']}\n"
            
    prompt += f"\nContent: {options.content}\nClass:"
    return prompt

def get_class_probability(
    content: str,
    classes: List[str],
    model: str,
    temperature: float,
    examples: Optional[List[Dict[str, str]]] = None,
    custom_prompt: Optional[str] = None
) -> Tuple[str, float]:
    try:
        options = ClassificationOptions(
            content=content,
            classes=classes,
            examples=examples
        )
        
        prompt = _create_prompt(options, custom_prompt)
        llm_model = llm.get_model(model)
        
        response = llm_model.prompt(
            prompt,
            temperature=temperature,
            top_logprobs=1
        )
        
        if isinstance(response.response_json, dict):
            if "choices" in response.response_json:
                # Completion model
                result = response.response_json["choices"][0]["text"].strip()
                prob = _extract_completion_logprobs(response.response_json)
            else:
                # Chat model
                result = response.response_json.get("content", "").strip()
                prob = _extract_chat_logprobs(response.response_json)
            return result, prob
            
        return response.text().strip(), 1.0
        
    except Exception as e:
        logger.error("Classification error: %s", e)
        return "Error", 0.0

def _make_serializable(obj):
    """Convert response object to JSON serializable format"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    return obj

@llm.hookimpl
def register_models(register):
    pass  # No custom models to register for this plugin

@llm.hookimpl
def register_prompts(register):
    pass  # No custom prompts to register for this plugin