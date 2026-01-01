import streamlit as st
from app.rag_chain import rag_chain

st.set_page_config(page_title="Scape Resident Assistant")
st.title("Scape Resident Assistant")

if "messages" not in st.session_state:
    st.session_state.messages=[]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
if prompt := st.chat_input("Ask about building rules...."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        response = rag_chain.invoke(prompt)
        st.markdown(response)
        
    st.session_state.messages.append({"role" : "assistant", "content" : response})