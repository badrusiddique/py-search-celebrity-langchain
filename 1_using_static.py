import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

 # Initialize OpenAI LLM with the provided key
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize the OpenAI LLM with a specific temperature default - 0.7
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.8)

# streamlit app setup
st.title("LangChain OpenAI: Demo Search Python App")
input_text = st.text_input("Enter a topic to search:")

if input_text:
    response = llm(input_text)  # Call the LLM with the input text

    # Display the response in the Streamlit app
    st.write("Response from OpenAI:")
    st.write(response)
