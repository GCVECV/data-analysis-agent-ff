import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

@dataclass
class LLMConfig:
    api_key: str = os.getenv("API_KEY", "")
    base_url: str = os.getenv("BASE_URL", "")
    model: str = os.getenv("MODEL", "deepseek-chat")
    temperature: float = 0.1

@dataclass
class AgentConfig:
    max_steps: int = 10
    max_tool_calls: int = 15
    max_steps_per_plan: int = 2
    context_token_budget: int = 16000
    tool_content_max_chars: int = 3000
    report_on_max_steps: bool = True
    max_reflects: int = 2
    replan_after_fails: int = 2
    max_replans: int = 2
    enable_memory: bool = True
    
    
    
    
        
