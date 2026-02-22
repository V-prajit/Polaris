import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_gemini():
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("Error: GEMINI_API_KEY not found in .env")
        return
        
    genai.configure(api_key=key)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        response = model.generate_content("Respond with exactly one word: 'Success'")
        print("Gemini API Test Response:", response.text.strip())
    except Exception as e:
        print("Error calling Gemini API:", str(e))

if __name__ == "__main__":
    test_gemini()
