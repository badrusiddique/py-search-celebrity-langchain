import os
import streamlit as st

from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = openai_key

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.8)

st.title("LangChain OpenAI: Celebrity Search App")
input_text = st.text_input("Enter a celebrity name to search:")

prompt = PromptTemplate(
    input_variables=["celebrity_name"],
    template="Search for information about Indian celebrity {celebrity_name} and provide a brief summary."
)

if input_text:
    # A chain in LangChain is made up of links
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(celebrity_name=input_text) # Call the LLM with the formatted prompt

    st.write("Response from OpenAI:")
    st.write(response)
