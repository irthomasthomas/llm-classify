import llm
from llm_classify import classify_content

# Ensure the model is loaded.  This avoids a race condition.
llm.get_model("gpt-4o")

content = ["This is a positive review.", "I am very disappointed.", "The food was okay."]
classes = ["positive", "negative", "neutral"]
model_name = "gpt-4o"
temperature = 0.0  # Use 0 for more deterministic results

results = classify_content(content, classes, model_name, temperature)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Class: {result['class']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")  # Safely access score
    print("---")

