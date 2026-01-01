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

template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
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

def save_to_file(question, answer):
    with open("scape_answer.txt", "a") as f:
        f.write(f"_____Query_____\n")
        f.write(f"Question: {question}\n")
        f.write(f"Question: {answer}\n")  
        
        
        

if __name__ == "__main__":
    question = "Can I use communal areas?"
    response = rag_chain.invoke(question)
    print("Answer:", response)
    save_to_file(question, response)


