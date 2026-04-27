import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Patch OpenAI
def mock_create(*args, **kwargs):
    messages = kwargs.get("messages", [])
    model = kwargs.get("model", "")
    
    mock_resp = MagicMock()
    mock_msg = MagicMock()
    
    msg_content = str(messages)
    
    if "Generate 5 diverse JSON specs" in msg_content:
        mock_msg.content = json.dumps({
            "specs": [
                {
                    "id": f"seed_{i}",
                    "system_prompt": "You are a test agent.",
                    "tools": ["web_search", "scrape_page"],
                    "planning_strategy": "react",
                    "stop_condition": {"min_report_length": 50, "must_include_citations": True, "max_steps": 3}
                } for i in range(1, 6)
            ]
        })
    elif "Evaluate the following research report" in msg_content:
        mock_msg.content = json.dumps({
            "completeness": 4, "accuracy": 4, "citations": 4, "coherence": 4, "feedback": "Good job."
        })
    elif "Does the source content support the claims" in msg_content:
        mock_msg.content = "YES"
    elif "Please perform deep research" in msg_content or "Task:" in msg_content:
        if len(messages) < 3:
            # Simulate a tool call
            mock_tool = MagicMock()
            mock_tool.id = "call_1"
            mock_tool.function.name = "python_repl"
            mock_tool.function.arguments = '{"code": "print(2+2)"}'
            mock_msg.tool_calls = [mock_tool]
            mock_msg.content = None
        else:
            mock_msg.content = "Final report with enough length and a citation: https://example.com"
            mock_msg.tool_calls = None
    elif "You are a planner" in msg_content:
        mock_msg.content = "1. Search\n2. Read\n3. Write"
    elif "fix a failing agent spec" in msg_content:
        mock_msg.content = json.dumps({
            "id": "mutated_1",
            "system_prompt": "I am a better agent.",
            "tools": ["python_repl"],
            "planning_strategy": "plan_and_solve",
            "stop_condition": {"min_report_length": 100, "must_include_citations": True, "max_steps": 5}
        })
    else:
        mock_msg.content = "Default."
        mock_msg.tool_calls = None

    def model_dump(exclude_unset=False):
        return {"role": "assistant", "content": mock_msg.content, "tool_calls": getattr(mock_msg, "tool_calls", None)}
    mock_msg.model_dump = model_dump
    
    mock_resp.choices = [MagicMock(message=mock_msg)]
    return mock_resp

class TestStemAgentLoop(unittest.TestCase):
    @patch('openai.resources.chat.completions.Completions.create', side_effect=mock_create)
    def test_main_loop(self, mock_openai):
        # Temporarily lower generations for test
        import yaml
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        
        orig_gens = cfg["evolution"]["generations"]
        cfg["evolution"]["generations"] = 2
        
        with open("config.yaml", "w") as f:
            yaml.dump(cfg, f)
            
        try:
            import main
            main.main()
        finally:
            # Restore config
            cfg["evolution"]["generations"] = orig_gens
            with open("config.yaml", "w") as f:
                yaml.dump(cfg, f)

if __name__ == '__main__':
    unittest.main()
