from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv;

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

video_id = "-HzgcbRXUK8"

loader = YoutubeLoader.from_youtube_url(
    f"https://www.youtube.com/watch?v={video_id}", 
    add_video_info=False,
    language=["en"]
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

split_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(split_docs, embeddings)

retriver = vector_store.as_retriever(search_type="similarity", kwargs={"k": 4})

parallel_chain = RunnableParallel({
    "context": retriver | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant. Based on the context provided, answer the question.
    if the context is insufficient, say "I don't know."
    Context: {context}
    Question: {question}
    """,
        input_variables=["context", "question"]
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

res = main_chain.invoke("Can you summarize the video?")
print(res)