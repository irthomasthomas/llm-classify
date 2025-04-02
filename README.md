# llm-classify

[![PyPI](https://img.shields.io/pypi/v/llm-classify.svg)](https://pypi.org/project/llm-classify/)
[![Changelog](https://img.shields.io/github/v/release/irthomasthomas/llm-classify?include_prereleases&label=changelog)](https://github.com/irthomasthomas/llm-classify/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/irthomasthomas/llm-classify/blob/main/LICENSE)

LLM plugin for content classification using various language models

## Table of Contents

- [llm-classify](#llm-classify)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Options](#options)
    - [Basic Examples](#basic-examples)
    - [Advanced Examples](#advanced-examples)
  - [Python API](#python-api)
  - [Development](#development)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-classify
```

## Usage

This plugin adds a new `classify` command to the LLM CLI. You can use it to classify content into predefined categories using various language models.

### Options

- `content`: The content(s) to classify. You can provide multiple items.
- `-c, --classes`: Class options for classification (at least two required).
- `-m, --model`: LLM model to use (default: gpt-3.5-turbo).
- `-t, --temperature`: Temperature for API call (default: 0).
- `-e, --examples`: Examples in the format 'content:class' (can be used multiple times).
- `-p, --prompt`: Custom prompt template.
- `--no-content`: Exclude content from the output.

You can also pipe content to classify:

```bash
echo "This is exciting news!" | llm classify -c positive -c negative -c neutral
```

### Basic Examples

1. Basic classification using a custom model and temperature:

```bash
llm classify "The weather is nice today" -c good -c bad -c neutral -m gpt-4 -t 0.7
```
Output:
```json
[
  {
    "class": "good",
    "score": 0.998019085206617,
    "content": "The weather is nice today"
  }
]
```

2. Basic classification multi-processing with default model:
```bash
llm classify "I love this product" "This is terrible" -c positive -c negative -c neutral
```
Output:
```json
[
  {
    "class": "positive",
    "score": 0.9985889762314736,
    "content": "I love this product"
  },
  {
    "class": "negative",
    "score": 0.9970504305526415,
    "content": "This is terrible"
  }
]
```

3. Providing examples for few-shot learning:

```bash
llm classify "The stock market crashed" \
-c economic -c political -c environmental \
-e "New trade deal signed:economic" -e "President gives speech:political" \
-e "Forest fires in California:environmental"
```

4. Using a custom prompt:

```bash
llm classify "Breaking news: Earthquake in Japan" \
-c urgent -c not-urgent \
-p "Classify the following news headline as either urgent or not-urgent:"
```

5. Multiple classification with Examples:
```bash
llm classify 'news.ycombinator.com' 'facebook.com' 'ai.meta.com' \
-c 'signal' -c 'noise' -c 'neutral' \
-e "github.com:signal" -e "arxiv.org:signal" -e "instagram.com:noise" \
-e "pinterest.com:noise" -e "anthropic.ai:signal" -e "twitter.com:noise" 
--model openrouter/openai/gpt-4-0314
```
```json
[{
   "class": "signal",
   "score": 0.9994780818067087,
   "content": "news.ycombinator.com"
},
{
   "class": "noise",
   "score": 0.9999876476902904,
   "content": "facebook.com"
},
{
   "class": "signal",
   "score": 0.9999895549275502,
   "content": "ai.meta.com"
}]
```

6. Terminal commands classification:
```bash
llm classify 'df -h' 'chown -R user:user /' \
-c 'safe' -c 'danger' -c 'neutral' \
-e "ls:safe" -e "rm:danger" -e "echo:neutral" \
--model gpt-4o-mini
```
```json
[{
   "class": "neutral",
   "score": 0.9995317830277939,
   "content": "df -h"
},
{
   "class": "danger",
   "score": 0.9964036839906633,
   "content": "chown -R user:user /"
}]
   ```

7. Classify a tweet
```shell
llm classify $tweet -c 'AI' -c 'ASI' -c 'AGI' --model gpt-4o-mini
```
```json
[{
   "class": "asi",
   "score": 0.9999984951481323,
   "content": "Superintelligence is within reach.
   Building safe superintelligence (SSI) is the most important technical problem of our time.
   We've started the worlds first straight-shot SSI lab, with one goal and one product:
   a safe superintelligence."
}]
```

8. Verify facts
```shell
llm classify "<source>$(curl -s docs.jina.ai)</source> \
<statement>Jina ai has an image generation api</statement>" \
-c True -c False --model gpt-4o --no-content
```
```json
[{
      "class": "false",
      "score": 0.99997334352929
}]
```

### Advanced Examples

1. **Acting on the classification result in a shell script:**
```bash
class-tweet() {
   local tweet="$1"
   local threshold=0.6
   local class="machine-learning"

   result=$(llm classify "$tweet" -c 'PROGRAMMING' -c 'MACHINE-LEARNING' \
   --model openrouter/openai/gpt-4o-mini \
   | jq -r --arg class "$class" --argjson threshold "$threshold" \
   '.[0] | select(.class == $class and .score > $threshold) | .class')

   if [ -n "$result" ]; then
      echo "Tweet classified as $class with high confidence. Executing demo..."
      echo "Demo: This is a highly relevant tweet about $class"
   else
      echo "Tweet does not meet classification criteria."
   fi
}
```
```
Tweet classified as machine-learning with high confidence. Executing demo...
Demo: This is a highly relevant tweet about machine-learning
```

2.  **Piping multiple lines using heredoc:**
```bash
cat <<EOF | llm classify -c 'tech' -c 'sports' -c 'politics'
AI makes rapid progress
Football season starts soon
New tax policy announced
EOF
```
```json
[{
   "class": "tech",
   "score": 0.9998246033937837,
   "content": "AI makes rapid progress"
},
{
   "class": "sports",
   "score": 0.999863096482142,
   "content": "Football season starts soon"
},
{
   "class": "politics",
   "score": 0.999994561441089,
   "content": "New tax policy announced"
}]
```

3.  **Parsing classification output with `jq`:**
```bash
echo "OpenAI releases GPT-4" | llm classify -c 'tech' -c 'business' | jq '.[0].class'
```

3.  **Simplifying output for shell scripts:**
```bash
echo "Breaking news: earthquake hits city" | llm classify -c 'world' -c 'local' \
-c 'sports' | jq -r '.[0].class'
```
```
world
```
```bash
if [[ $(echo "Breaking news: earthquake hits city" | llm classify -c 'world' \
-c 'local' -c 'sports' | jq -r '.[0].class') == "world" ]]; then
   echo "This is world news"
fi
```
```
This is world news
```
## Python API

1.  **Automated Email Sorter with Confidence Threshold**

Sort incoming emails into categories ("Urgent", "Work", "Personal", "Spam"), using a confidence threshold to automate moving emails or flagging them for review.

```python
from llm_classify import classify_content

emails = [
    "The server is down and needs immediate attention.",
    "Meeting tomorrow at 2 PM.",
    "Your package has been shipped.",
    "Make money fast! Click here now."
]
classes = ["Urgent", "Work", "Personal", "Spam"]

results = classify_content(emails, classes, model="gpt-4o-mini", temperature=0.7)

for result in results:
    if result["class"] == "Urgent" and result["confidence"] > 0.9:
        print(f"Automatically moving: {result['content']} to Urgent folder")
    elif result["class"] == "Spam" and result["confidence"] > 0.85:
        print(f"Moving {result['content']} to Spam folder")
    else:
        print(f"Flagging {result['content']} for manual review")
```

2. **Shell Command Safety Interceptor**

Classify shell commands based on safety, blocking potentially dangerous commands based on confidence.

```python
from llm_classify import classify_content

dangerous_commands = ["rm -rf /", "chmod 777 /etc", "curl http://malicious"]

results = classify_content(
    dangerous_commands,
    classes=["safe", "risky"],
    model="claude-3.5-sonnet",
    examples=[
        "chmod 000 /etc/passwd:risky",
        "ls -l:safe"
    ],
    custom_prompt="Classify command danger level considering system-wide impact:"
)

for cmd, result in zip(dangerous_commands, results):
    if result['class'] == 'risky' and result.get('confidence', 0) > 0.85:
        print(f"Blocking dangerous command: {cmd}")
        # Add iptables rule to block origin IP
    elif result['class'] == 'risky':
        print(f"Flagging medium-risk command ({result['confidence']}): {cmd}")
```

3.  **System Log Anomaly Detection and Confidence-Based Alerting**

Classify system log messages by severity to trigger alerts dynamically based on confidence levels.

```python
from llm_classify import classify_content

log_messages = [
    "INFO: User logged in successfully.",
    "WARNING: Disk space nearing capacity.",
    "ERROR: Database connection timeout.",
    "CRITICAL: System core dump detected!"
]
classes = ["info", "warning", "error", "critical"]

results = classify_content(log_messages, classes, model="gpt-4o-mini")

for log, result in zip(log_messages, results):
    confidence = result['confidence']
    if result['class'] == 'critical' and confidence > 0.95:
        print(f"CRITICAL ERROR: '{log}'. Triggering immediate escalation and system alert!")
        # Trigger system-wide alert and escalation protocols
    elif result['class'] == 'error' and confidence > 0.8:
        print(f"ERROR: '{log}'. Sending notification to operations team.")
        # Send notification to relevant team
    elif result['class'] == 'warning' and confidence > 0.7:
        print(f"WARNING: '{log}'. Logging for review.")
        # Log warning for later review
    else:
        print(f"INFO: '{log}'. Standard logging.")
        # Standard info logging
```

4. **AI Model Code Snippet Classification for Automated Code Review**

Classify Python code snippets related to Machine Learning tasks. Confidence scores determine the level of automated code review applied.

```python
from llm_classify import classify_content

code_snippets = [
    "import pandas as pd\ndata = pd.read_csv('data.csv')",
    "model = RandomForestClassifier()\nmodel.fit(X_train, y_train)",
    "accuracy = accuracy_score(y_test, predictions)"
]
classes = ["data_loading", "model_training", "evaluation", "other"]

results = classify_content(code_snippets, classes, model="gpt-4o-mini")

for snippet, result in zip(code_snippets, results):
    if result['class'] == 'data_loading' and result['confidence'] > 0.8:
        print(f"Data Loading Snippet: '{snippet[:50]}...'. Initiating automated data quality checks.")
        # Here, you'd trigger automated data validation scripts
    elif result['confidence'] < 0.6:
        print(f"Low confidence for: '{snippet[:50]}...'. Flagging for manual code review.")
    else:
        print(f"Snippet classified as '{result['class']}': '{snippet[:50]}...'. Proceeding with standard code review.")
```

5.  **Python Function Type Classification for Automated Documentation Generation**

Classify Python functions based on their purpose to automate documentation generation. Confidence guides the automation level: high confidence triggers auto-documentation; low confidence prompts manual documentation.

```python
from llm_classify import classify_content

functions = [
    """def calculate_average(data_list): return sum(data_list) / len(data_list)""",
    """def fetch_user_profile(user_id): api_call('/users/' + user_id)""",
    """def render_ui_button(text, onclick_action): # UI rendering logic here""",
    """def _internal_utility_function(param): # ... """
]
classes = ["data_processing", "api_interaction", "ui_rendering", "utility", "unknown"]

results = classify_content(functions, classes, model="gpt-4o-mini")

for function_code, result in zip(functions, results):
    confidence = result['confidence']
    if result['class'] in ['data_processing', 'utility'] and confidence > 0.85:
        print(f"Function classified as '{result['class']}': '{function_code[:50]}...'. Auto-generating documentation.")
    elif confidence < 0.7:
        print(f"Unclear function type for: '{function_code[:50]}...'. Manual documentation recommended.")
    else:
        print(f"Function classified as '{result['class']}': '{function_code[:50]}...'. Standard documentation workflow.")
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
cd llm-classify
python -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
llm install -e '.[test]'
```

To run the tests:

```bash
pytest
```

## Contributing

Contributions to llm-classify are welcome! Please refer to the [GitHub repository](https://github.com/irthomasthomas/llm-classify) for more information on how to contribute.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.