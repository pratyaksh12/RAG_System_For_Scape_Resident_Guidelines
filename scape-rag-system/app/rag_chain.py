from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

DB_PATH = "app/db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# create the retriever
retriever = vector_store.as_retriever(search_kwargs={"k" : 5})

# define the prompt

template = """You are a helpful Scape Resident Assistant.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I cannot find this information in the resident guidelines."
Do not make up rules.
Context:
{context}
Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# build the chain
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


rag_chain = (
    {"context" : retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    response = rag_chain.invoke("Will I be responsible for false fire alarm trigger?")
    print("Answer:", response)


