"""
Here you will find all the agents form create_agent, already with their prompts loaded
To edit prompts, check the prompts.yaml document
"""

from functions import read_folder, read_template, save_initial_persona, retriever_tool, read_document
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
undefined_clients_agent_prompt = prompts["undefined_clients_agent_prompt"]
life_coach_assistant_prompt = prompts["life_coach_assistant_prompt"]


session_agent = create_agent(
    system_prompt=session_agent_prompt,
    model="anthropic:claude-haiku-4-5",
    tools=[read_folder, read_template],  # verify/save handled by Python orchestrator
)

undefined_clients_agent = create_agent(
    system_prompt=undefined_clients_agent_prompt,
    model="openai:gpt-5-mini",
    tools=[read_folder, save_initial_persona],
)


life_coach_assistant_agent = create_agent(
    system_prompt=life_coach_assistant_prompt,
    model="openai:gpt-5-mini",
    tools=[read_folder, retriever_tool, read_document]
)

#google_genai:gemini-2.5-flash