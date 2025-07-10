from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm=GoogleGenerativeAI(model="gemini-2.0-flash")
output=llm.invoke("自己紹介をしてください")
print(output)