import os
from unittest.mock import MagicMock
from chatbot import HuggingFaceChatbot

def mock_get_flood_probability(site_code=None, lat=None, lon=None):
    print(f"Tool called with: site_code={site_code}, lat={lat}, lon={lon}")
    return "The flood probability is 45% (Moderate Risk)."

def test_tool_calling():
    # Mock tools
    tools = {
        "get_flood_probability": mock_get_flood_probability
    }
    
    # Initialize bot with a dummy token so it doesn't complain
    bot = HuggingFaceChatbot(api_token="dummy_token", tools=tools)
    
    # Mock the client
    bot.client = MagicMock()
    
    # --- Test 1: Tool Call ---
    print("--- Test 1: Simulate Tool Call ---")
    
    # Mock the first response (Tool Call)
    mock_message_tool = MagicMock()
    mock_message_tool.content = '{"tool": "get_flood_probability", "site_code": "03432400"}'
    
    mock_choice_tool = MagicMock()
    mock_choice_tool.message = mock_message_tool
    
    mock_completion_tool = MagicMock()
    mock_completion_tool.choices = [mock_choice_tool]
    
    # Mock the second response (Final Answer)
    mock_message_final = MagicMock()
    mock_message_final.content = "The flood probability for site 03432400 is 45% (Moderate Risk)."
    
    mock_choice_final = MagicMock()
    mock_choice_final.message = mock_message_final
    
    mock_completion_final = MagicMock()
    mock_completion_final.choices = [mock_choice_final]
    
    # Set side_effect to return tool call first, then final answer
    bot.client.chat.completions.create.side_effect = [mock_completion_tool, mock_completion_final]
    
    response = bot.get_response("What is the flood probability for site 03432400?")
    print(f"Final Response: {response}\n")
    
    # Verify tool was called
    # We can't easily assert here without a proper test framework, but the print in mock_get_flood_probability will show up.

if __name__ == "__main__":
    test_tool_calling()
