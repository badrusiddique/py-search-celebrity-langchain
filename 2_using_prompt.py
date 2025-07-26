import os
import streamlit as st

from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

 # Initialize OpenAI LLM with the provided key
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize the OpenAI LLM with a specific temperature default - 0.7
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.8)

# streamlit app setup
st.title("LangChain OpenAI: Celebrity Search App")
input_text = st.text_input("Enter a celebrity name to search:")

# Define the prompt template for searching celebrity information
prompt = PromptTemplate(
    input_variables=["celebrity_name"],
    template="Search for information about Indian celebrity {celebrity_name} and provide a brief summary."
)

if input_text:
    response = llm(prompt.format(celebrity_name=input_text))  # Call the LLM with the formatted prompt

    # Display the response in the Streamlit app
    st.write("Response from OpenAI:")
    st.write(response)
