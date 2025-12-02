# TODO - Create an agent
# TODO - Pull the data from a DB like Google Sheets
# TODO Tool to READ the DB
# TODO 

from langchain.agents import create_agent
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

def load_prompts():
    with open("prompts.yaml", "r", encoding="UTF-8") as f:
        prompts = yaml.safe_load(f)
    return prompts

prompts = load_prompts()
first_draft_agent_prompt = prompts["class0_prompt"]


first_draft_agent = create_agent(
    system_prompt=first_draft_agent_prompt,
    model="google_genai:gemini-2.5-flash"
)

response = first_draft_agent.invoke({
    "role" : "user",
    "messages" : "Hello, what can you do?"
})

for i,msg in enumerate(response["messages"]):
    msg.pretty_print()