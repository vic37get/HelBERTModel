from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
chave_api = os.getenv('TOKEN_API')

# mensagens = [
#     SystemMessage("Traduza o texto a seguir para o inglês"),
#     HumanMessage("Eu gosto de estudar")
# ]

modelo = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", api_key=chave_api)
parser = StrOutputParser()

chain = modelo | parser

template_messagem = ChatPromptTemplate.from_messages([
    ("system", "Traduza o texto a seguir para {language} por completo. Desejo que o texto esteja totalmente traduzido!"),
    ("user", "{texto}")
])

chain = template_messagem | modelo | parser

texto = chain.invoke({"language": "inglês", "texto": "Eu quero aprender."})

print(texto)