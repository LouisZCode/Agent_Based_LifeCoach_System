"""
Here you will find all the agents form create_agent, already with their prompts loaded
To edit prompts, check the prompts.yaml document
"""

from functions import read_folder, read_template, save_summary, save_homework, save_session_draft
from langchain.agents import create_agent
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

def load_prompts():
    with open("agents/prompts.yaml", "r", encoding="UTF-8") as f:
        prompts = yaml.safe_load(f)
    return prompts

prompts = load_prompts()
session_agent_prompt = prompts["session_agent_prompt"]


session_agent = create_agent(
    system_prompt=session_agent_prompt,
    model="anthropic:claude-haiku-4-5",
    tools=[read_folder, read_template, save_summary, save_session_draft, save_homework],
)

"""
discovery_agent = create_agent(
    system_prompt=first_draft_agent_prompt,
    model="google_genai:gemini-2.5-flash",
    tools=[list_folder_names, read_client_folder]
)

chat_agent = create_agent(
    system_prompt=first_draft_agent_prompt,
    model="google_genai:gemini-2.5-flash",
    tools=[list_folder_names, read_client_folder]
)
"""