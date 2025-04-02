import os
import pathlib
import llm
import click
import json
import math
import time
import sys
import logging
import string
import re
from datetime import datetime
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
            "datetime_utc": str,  # Human-readable timestamp in ISO format
            "response_json": str,
            "confidence": float
        }, pk="id")
    else:
        # Ensure columns exist in case of outdated schema
        table = db["classifications"]
        if "confidence" not in table.columns_dict:
            table.add_column("confidence", float)
        if "datetime_utc" not in table.columns_dict:
            table.add_column("datetime_utc", str)
        # Remove timestamp if exists (we're not concerned with backward compatibility)
        if "timestamp" in table.columns_dict:
            try:
                db.execute("ALTER TABLE classifications DROP COLUMN timestamp")
            except Exception as e:
                logger.warning(f"Failed to drop timestamp column: {e}")
    
    return db

def process_classification_response(response, classes: List[str]) -> Tuple[Optional[str], float]:
    try:
        response_text = (response.text() or "").strip().lower()
        valid_letters = [chr(97+i) for i in range(min(len(classes), 26))]
        
        # Find first valid letter using regex
        match = re.search(rf'\b([{"".join(valid_letters)}])\b', response_text)
        selected_letter = match.group(1) if match else None
        
        if not selected_letter or selected_letter not in valid_letters:
            return None, 0.0

        confidence = 0.0
        try:
            response_json = response.json()
            if 'logprobs' in response_json:
                for token in response_json['logprobs']['content']:
                    # Support both dict and object tokens
                    token_value = token['token'] if isinstance(token, dict) else token.token
                    token_logprob = token['logprob'] if isinstance(token, dict) else token.logprob
                    if token_value.lower() == selected_letter:
                        confidence = math.exp(token_logprob)
                        break
        except (AttributeError, KeyError, TypeError) as e:
            logger.error(f"Logprob parsing error: {str(e)}")
        
        try:
            class_index = ord(selected_letter) - 97
            if 0 <= class_index < len(classes):
                return classes[class_index], round(confidence, 4)
        except IndexError:
            pass
            
        return None, 0.0

    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return None, 0.0

def format_prompt(content: str, classes: Tuple[str], examples_list: Optional[List[Dict[str, str]]], custom_prompt: Optional[str]) -> str:
    # Assign letters to classes with validation
    class_options = []
    for i, cls in enumerate(classes):
        if i >= 26:
            logger.warning("Too many classes for letter mapping (>26)")
            break
        class_options.append(f"{chr(97+i)}) {cls}")
    class_options_str = "\n".join(class_options)
    
    default_prompt = f"""<INSTRUCTIONS>
Your task is to classify the content into ONE category using the letter corresponding to your choice.

AVAILABLE CLASSES:
{class_options_str}

CONTENT TO CLASSIFY:
{{content}}

Respond ONLY with the single corresponding lowercase letter with no punctuation.</INSTRUCTIONS>"""
    prompt_template = custom_prompt or default_prompt
    formatted_prompt = prompt_template.format(content=content)

    if examples_list:
        formatted_prompt += "\n\nEXAMPLES:"
        for example in examples_list:
            try:
                example_idx = list(classes).index(example['class'])
                example_letter = chr(97 + example_idx)
                formatted_prompt += f"\nContent: {example['content']}\nResponse: {example_letter}"
            except ValueError:
                logger.warning(f"Skipping example with invalid class: {example['class']}")
    return formatted_prompt

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("content", type=str, required=False, nargs=-1)
    @click.option("-c", "--classes", required=True, multiple=True, help="Class options for classification")
    @click.option("-m", "--model", default="gpt-4o-mini", help="LLM model to use")
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
        
        db = setup_database(llm.user_dir() / "logs.db")
    
        results = []
        for item in content:
            try:
                formatted_prompt = format_prompt(item, classes, examples_list, prompt)

                response = llm.get_model(model).prompt(
                    formatted_prompt,
                    temperature=temperature,
                    top_logprobs=1,
                    max_tokens=1  # Add max_tokens limit
                )
                
                evaluated_response = str(response)

                result, confidence = process_classification_response(response, list(classes))
                
                if result:
                    serializable_response = {}
                    try:
                        serializable_response = _make_serializable(response.json())
                    except (AttributeError, TypeError):
                        pass
                    
                    # Get current time in ISO format
                    current_datetime_utc = datetime.utcnow().isoformat() + "+00:00"
                    
                    db["classifications"].insert({
                        "content": item if not no_content else "",
                        "class": result,
                        "model": model,
                        "temperature": temperature,
                        "datetime_utc": current_datetime_utc,
                        "response_json": json.dumps(serializable_response),
                        "confidence": confidence
                    })
                    
                    result_dict = {
                        "class": result,
                        "confidence": confidence
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
    # Create/setup the database
    db = setup_database(llm.user_dir() / "logs.db")
    
    results = []
    for item in content:
        winner, probability = get_class_probability(
            item, classes, model, temperature, examples, custom_prompt
        )
        
        # Get current time in ISO format
        current_datetime_utc = datetime.utcnow().isoformat() + "+00:00"
        
        # Save to database
        db["classifications"].insert({
            "content": item if not no_content else "",
            "class": winner,
            "model": model,
            "temperature": temperature,
            "datetime_utc": current_datetime_utc,
            "response_json": "{}",  # No response_json available here
            "confidence": probability
        })
        
        result = {
            "class": winner, 
            "score": probability,
            "datetime_utc": current_datetime_utc
        }
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
        
        try:
            response_json = response.json()
            if isinstance(response_json, dict):
                if "choices" in response_json:
                    # Completion model
                    result = response_json["choices"][0]["text"].strip()
                    prob = _extract_completion_logprobs(response_json)
                else:
                    # Chat model
                    result = response_json.get("content", "").strip()
                    prob = _extract_chat_logprobs(response_json)
                return result, prob
        except (AttributeError, TypeError):
            pass
            
        return response.text().strip(), 1.0
        
    except Exception as e:
        logger.error("Classification error: %s", e)
        return "Error", 0.0

def _make_serializable(data):
    if isinstance(data, dict):
        return {k: _make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_make_serializable(item) for item in data]
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    else:
        return str(data)  # Convert non-serializable objects to string

@llm.hookimpl
def register_models(register):
    pass  # No custom models to register for this plugin

@llm.hookimpl
def register_prompts(register):
    pass  # No custom prompts to register for this plugin