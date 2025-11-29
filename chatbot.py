import os
from openai import OpenAI

class HuggingFaceChatbot:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct:novita", api_token=None):
        """
        Initialize the HuggingFace Chatbot using the OpenAI client.
        
        Args:
            model_id (str): The HuggingFace model ID to use. 
                            Defaults to 'meta-llama/Llama-3.1-8B-Instruct:novita'.
            api_token (str): Optional HuggingFace API token.
        """
        self.model_id = model_id
        
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
            "Keep answers concise and helpful."
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
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error connecting to chatbot: {str(e)}"

if __name__ == "__main__":
    # Simple test
    # Ensure HF_TOKEN is set in env for this test to work
    bot = HuggingFaceChatbot()
    print(bot.get_response("What is a flash flood?"))
