from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv;

load_dotenv()

# step: 1 load transcript
video_id = "-HzgcbRXUK8"
loader = YoutubeLoader.from_youtube_url(
    f"https://www.youtube.com/watch?v={video_id}", 
    add_video_info=False,
    language=["en"]
)

docs = loader.load()

# step: 2 split documents
recursiveTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = recursiveTextSplitter.split_documents(docs)

# step: 3 create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#step 4: vector store
vector_store = FAISS.from_documents(split_docs, embeddings)

#step 5: retriver
retriver = vector_store.as_retriever(search_type="similarity", kwargs={"k": 4})

question = "what is the capital of France?"

fetched_docs = retriver.invoke(question)

#augmentation
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant. Answer only from the provided transcript context, 
    if the context is insufficient just say you dont know.

    Context:
    {context}

    Question:
    {question}
    """,
    input_variables=["context", "question"]
)

context_text = "\n\n".join(doc.page_content for doc in fetched_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})

res = llm.invoke(final_prompt)

print(res.content)