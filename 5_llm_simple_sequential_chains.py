import os
import streamlit as st

from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = openai_key

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.8)

st.title("LangChain OpenAI: Celebrity Search App")
st.markdown("Enter an Indian celebrity's name to get details from OpenAI's GPT-3.5-turbo-instruct model.")

prompt_name = PromptTemplate(
    input_variables=["celebrity_name"],
    template="Give me the full legal name of Indian celebrity {celebrity_name}."
)
llm_chain_name = LLMChain(llm=llm, prompt=prompt_name, output_key="celebrity_f_name")

prompt_dob = PromptTemplate(
    input_variables=["celebrity_f_name"],
    template="Now share me the date of birth of {celebrity_f_name}."
)
llm_chain_dob = LLMChain(llm=llm, prompt=prompt_dob, output_key="celebrity_dob")

prompt_age = PromptTemplate(
    input_variables=["celebrity_dob"],
    template="Now calculate the age as of today {celebrity_dob} and only print Full Name, DOB with Age."
)
llm_chain_age = LLMChain(llm=llm, prompt=prompt_age)

input_text = st.text_input("Enter a celebrity name to search:", placeholder="e.g., Shah Rukh Khan")
if input_text:
    # which are chains that execute their links in a predefined order
    # where each step has a single input/output, and the output of one step is the input to the next
    # only the last chain's output is returned
    sequential_chain = SimpleSequentialChain(chains=[llm_chain_name, llm_chain_dob, llm_chain_age])
    response = sequential_chain.run(input_text)

    st.markdown("**Response from OpenAI:**")
    st.write(response)

st.markdown("*Powered by OpenAI gpt-3.5-turbo-instruct via LangChain*")
