import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-3.0-flash')

response = model.generate_content("What is 2 raised to the power of 10?")
print(response.text)