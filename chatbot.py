import os
from openai import OpenAI

class HuggingFaceChatbot:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct:novita", api_token=None, tools=None):
        """
        Initialize the HuggingFace Chatbot using the OpenAI client.
        
        Args:
            model_id (str): The HuggingFace model ID to use. 
                            Defaults to 'meta-llama/Llama-3.1-8B-Instruct:novita'.
            api_token (str): Optional HuggingFace API token.
            tools (dict): Optional dictionary of tool functions. 
                          Format: {"tool_name": function_reference}
        """
        self.model_id = model_id
        self.tools = tools or {}
        
        # Use provided token or fall back to environment variable
        token = api_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        
        if not token:
            # We can't initialize the client properly without a token for this endpoint
            self.client = None
        else:
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=token,
            )

    def get_response(self, user_input, history=None):
        """
        Generate a response from the chatbot.
        
        Args:
            user_input (str): The user's message.
            history (list): List of previous messages (optional, for context).
                            Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        
        Returns:
            str: The chatbot's response.
        """
        if not self.client:
            return "Error: API Token is missing. Please provide a HuggingFace API Token."

        # Construct messages list
        messages = []
        
        # System prompt
        system_prompt = (
            "You are a helpful assistant for a Flash Flood Prediction Application. "
            "The app helps users check flood probability at USGS sites based on streamflow data. "
            "Answer questions about floods, safety, and how to interpret risk levels (Low < 30%, Moderate < 70%, High >= 70%). "
            "Keep answers concise and helpful.\n\n"
            "TOOLS AVAILABLE:\n"
            "If the user asks for a flood prediction or probability at a specific location, "
            "you can use the 'get_flood_probability' tool. "
            "To use a tool, your response must be ONLY a JSON object in this format:\n"
            '{"tool": "get_flood_probability", "site_code": "12345678"}\n'
            "OR if coordinates are provided:\n"
            '{"tool": "get_flood_probability", "lat": 36.16, "lon": -86.78}\n'
            "OR if a site name or general location is provided:\n"
            '{"tool": "get_flood_probability", "site_name": "Beaver Creek"}\n'
            "Do not add any other text when using a tool."
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # History
        if history:
            # Ensure history format matches OpenAI expectations (role/content)
            # Our app stores history as {"role": "user"/"assistant", "content": "..."} which is compatible
            messages.extend(history)
        
        # Current user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=500,
            )
            response_content = completion.choices[0].message.content
            
            # Check for tool usage
            import json
            import re
            
            try:
                # Look for JSON object in the response
                # Matches { "tool": ... } allowing for whitespace and newlines
                # We use a non-greedy match for the content inside braces, but we need to be careful about nested braces.
                # Since our tool calls are simple, a simple regex might suffice, or we can try to find the first '{' and last '}'
                
                # Regex to find a JSON-like block containing "tool":
                # This is a heuristic: find a block starting with { and containing "tool"
                match = re.search(r'(\{.*?"tool":.*?\})', response_content, re.DOTALL)
                
                if match:
                    json_str = match.group(1)
                    try:
                        tool_data = json.loads(json_str)
                        tool_name = tool_data.get("tool")
                        
                        if tool_name in self.tools:
                            # Execute tool
                            tool_func = self.tools[tool_name]
                            
                            # Extract args
                            kwargs = {k: v for k, v in tool_data.items() if k != "tool"}
                            
                            # Call function
                            tool_result = tool_func(**kwargs)
                            
                            # Append result to messages and ask for final response
                            messages.append({"role": "assistant", "content": response_content})
                            messages.append({"role": "system", "content": f"Tool Output: {tool_result}. Now answer the user's question based on this output."})
                            
                            # Second call to get final answer
                            final_completion = self.client.chat.completions.create(
                                model=self.model_id,
                                messages=messages,
                                max_tokens=500,
                            )
                            return final_completion.choices[0].message.content
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                return f"Error executing tool: {str(e)}"

            return response_content
        except Exception as e:
            return f"Error connecting to chatbot: {str(e)}"

if __name__ == "__main__":
    # Simple test
    # Ensure HF_TOKEN is set in env for this test to work
    bot = HuggingFaceChatbot()
    print(bot.get_response("What is a flash flood?"))
