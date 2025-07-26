import os
import streamlit as st

from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = openai_key

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.8)

st.title("LangChain OpenAI: Celebrity Search App")
st.markdown("Enter an Indian celebrity's name to get details from OpenAI's GPT-3.5-turbo-instruct model.")

prompt_name = PromptTemplate(
    input_variables=["celebrity_name"],
    template="Give me the full legal name of Indian celebrity {celebrity_name}."
)
celebrity_name_memory = ConversationBufferMemory(input_key="celebrity_name", memory_key="name_memory")
llm_chain_name = LLMChain(llm=llm, prompt=prompt_name, output_key="celebrity_f_name", memory=celebrity_name_memory)

prompt_dob = PromptTemplate(
    input_variables=["celebrity_f_name"],
    template="Share the date of birth of {celebrity_f_name} in the format DD-MM-YYYY."
)
celebrity_dob_memory = ConversationBufferMemory(input_key="celebrity_f_name", memory_key="dob_memory")
llm_chain_dob = LLMChain(llm=llm, prompt=prompt_dob, output_key="celebrity_dob", memory=celebrity_dob_memory)

prompt_age = PromptTemplate(
    input_variables=["celebrity_dob"],
    template="List up to three other celebrities born on {celebrity_dob} (in DD-MM-YYYY format) around the world, or state if none are known."
)
celebrities_on_dob_memory = ConversationBufferMemory(input_key="celebrity_dob", memory_key="on_dob_memory")
llm_chain_age = LLMChain(llm=llm, prompt=prompt_age, output_key="celebrities_on_dob", memory=celebrities_on_dob_memory)

input_text = st.text_input("Enter a celebrity name to search:", placeholder="e.g., Shah Rukh Khan")
if input_text:
    # which are chains that execute their links in a predefined order
    # where each step has a single input/output, and the output of one step is the input to the next
    # all the chains are executed sequentially with JSON input/output
    sequential_chain = SequentialChain(
        chains=[llm_chain_name, llm_chain_dob, llm_chain_age],
        input_variables=["celebrity_name"],
        output_variables=["celebrity_f_name", "celebrity_dob", "celebrities_on_dob"])
    response = sequential_chain({'celebrity_name': input_text})

    st.markdown("**Response from OpenAI:**")
    st.write(f"**Full Name**: {response['celebrity_f_name']}")
    st.write(f"**Date of Birth**: {response['celebrity_dob']}")
    st.write(f"**Other Celebrities Born on Same Date**: {response['celebrities_on_dob']}")

    with st.expander("**Celebrity Name Memory:**"):
        st.write(celebrity_name_memory.load_memory_variables({}))
    with st.expander("**Celebrity DOB Memory:**"):
        st.write(celebrity_dob_memory.load_memory_variables({}))
    with st.expander("**Celebrities on DOB Memory:**"):
        st.write(celebrities_on_dob_memory.load_memory_variables({}))

st.markdown("*Powered by OpenAI gpt-3.5-turbo-instruct via LangChain*")
