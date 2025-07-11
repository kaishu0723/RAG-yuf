from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pickle
from langchain.retrievers import EnsembleRetriever

query="温泉のある宿泊施設を教えてください"

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore=Chroma(
    embedding_function=embeddings,
    persist_directory="./vectorstore",
)
with open('./bm25.pkl','rb') as f:
    bm_data=pickle.load(f)


vector_ret=vectorstore.as_retriever(query=query)

ensemble_ret=EnsembleRetriever(
    retrievers=[vector_ret,bm_data],
    weights=[0.5,0.5]
)
ans=ensemble_ret.invoke(query)

print(type(ans))
print(len(ans))
for i in ans:
    print(i)