from langserve import RemoteRunnable

chain_remota = RemoteRunnable("http://localhost:8000/tradutor")
texto = chain_remota.invoke({"language": "inglês", "texto": "Eu consegui aprender os primeiros passos com LangChain."})
print(texto)