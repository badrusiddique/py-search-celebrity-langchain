import os
import streamlit as st

from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = openai_key

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.8)

st.title("LangChain OpenAI: Celebrity Search App")
st.markdown("Enter an Indian celebrity's name to get a one-line summary from OpenAI's GPT-3.5-turbo-instruct model.")

prompt = PromptTemplate(
    input_variables=["celebrity_name"],
    template="Search for information about Indian celebrity {celebrity_name} and provide a one-line brief summary."
)

input_text = st.text_input("Enter a celebrity name to search:", placeholder="e.g., Shah Rukh Khan")
if input_text:
    # Chain supports taking a BaseMemory object as its memory argument, allowing Chain object to persist data across multiple calls.
    # In other words, it makes Chain a stateful object.
    llm_chain_memory = LLMChain(llm=llm, prompt=prompt, memory=ConversationBufferMemory())
    response = llm_chain_memory.run(celebrity_name=input_text)

    st.markdown("**Response from OpenAI:**")
    st.write(response)

    # Example of using the chain with a hardcoded input - use context from previous runs
    response = llm_chain_memory.run(celebrity_name="Virat Kohli")

    st.markdown("**Response from OpenAI:**")
    st.write(response)

st.markdown("*Powered by OpenAI gpt-3.5-turbo-instruct via LangChain*")
