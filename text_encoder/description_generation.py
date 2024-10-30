import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from secret_key import API_tog_1 as API_KEY
import json

# Set TogetherAI API key
os.environ["TOGETHER_API_KEY"] = API_KEY


# Function to create a prompt and call the TogetherAI model
def generate_expanded_definition(definitions_dict):
    # Initialize the TogetherAI model
    model = ChatOpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=os.environ["TOGETHER_API_KEY"],
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    # Dictionary to store the expanded definitions
    expanded_definitions = {}

    # Iterate through each name and definition in the input dictionary
    for name, definition in definitions_dict.items():
        prompt = f"{definition} is the definition of the {name}. Please rewrite and expand this definition to make it more detailed and consistent with scientific fact. Briefness is required, using only one paragraph."

        # Send the prompt to the model
        messages = [
            SystemMessage(content="Please rewrite and expand the given definition."),
            HumanMessage(content=prompt),
        ]

        # Invoke the model and store the expanded definition (content) in the output dictionary
        response = model.invoke(messages)
        expanded_definitions[name] = response.content

    return expanded_definitions

def save_definitions(definitions, save_path='./text_encoder/data/text_expanded_definitions.json'):
    with open(save_path, 'w') as f:
        json.dump(definitions, f)
    return save_path

if __name__ == "__main__":
    # Example input dictionary
    with open("./text_encoder/data/text_definitions.json", "r") as file:
        input_definitions = json.load(file)

    # Get the expanded definitions
    output_definitions = generate_expanded_definition(input_definitions)
    path = save_definitions(output_definitions)
    # Print the results
    print(f"Generated expanded definitions:saved at {path}")