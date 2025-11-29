from chatbot import HuggingFaceChatbot

def test_chatbot():
    print("Testing HuggingFace Chatbot...")
    bot = HuggingFaceChatbot()
    
    questions = [
        "What is a flash flood?",
        "What should I do if I see a flood warning?",
        "How do I interpret a 80% flood probability?"
    ]
    
    for q in questions:
        print(f"\nUser: {q}")
        response = bot.get_response(q)
        print(f"Assistant: {response}")
        
        if "Error" in response:
            print("❌ Test Failed (API Error)")
        else:
            print("✅ Test Passed (Response received)")

if __name__ == "__main__":
    test_chatbot()
