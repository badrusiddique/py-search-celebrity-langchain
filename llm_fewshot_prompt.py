import os
import streamlit as st

from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"] = openai_key

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.8)

st.title("LangChain OpenAI: Celebrity Search App")
st.markdown("Enter an Indian celebrity's name to get details from OpenAI's GPT-3.5-turbo-instruct model.")

pre_examples = [
    {
        "celebrity_name": "Shah Rukh Khan",
        "place_of_birth": "New Delhi, India",
        "profession": "Actor, Producer",
        "notable_work": "Dilwale Dulhania Le Jayenge"
        },
    {
        "celebrity_name": "Priyanka Chopra",
        "place_of_birth": "Jamshedpur, India",
        "profession": "Actress, Singer",
        "notable_work": "Bajirao Mastani"
    },
    {
        "celebrity_name": "Amitabh Bachchan",
        "place_of_birth": "Allahabad, India",
        "profession": "Actor, Film Producer",
        "notable_work": "Sholay"
    },
    {
        "celebrity_name": "MS Dhoni",
        "place_of_birth": "Ranchi, India",
        "profession": "Cricketer",
        "notable_work": "Indian National Cricket Team Captain"
    }
]

prompt_template = PromptTemplate(
    input_variables=["celebrity_name", "place_of_birth", "profession", "notable_work"],
    template="Celebrity Name: {celebrity_name}\n Place of Birth: {place_of_birth}\n Profession: {profession}\n Notable Work: {notable_work}\n\n"
)

# print("Prompt Format:", prompt_template.format(
#     celebrity_name="Shah Rukh Khan",
#     place_of_birth="New Delhi, India",
#     profession="Actor, Producer",
#     notable_work="Dilwale Dulhania Le Jayenge"
# ))


# Create a FewShotPromptTemplate with the pre-defined examples
# This will help the model understand the format and context better
# and generate more accurate responses based on the provided examples
few_shot_prompt_name = FewShotPromptTemplate(
    examples=pre_examples,
    example_prompt=prompt_template,
    example_separator="\n\n",
    input_variables=["celebrity_name"],
    suffix="Now, provide the details for the celebrity you are searching for: {celebrity_name}.",
)

# print("Prompt Format:", few_shot_prompt_name.format(
#     celebrity_name="Shah Rukh Khan"
# ))

llm_chain = LLMChain(llm=llm, prompt=few_shot_prompt_name)


input_text = st.text_input("Enter a celebrity name to search:", placeholder="e.g., Shah Rukh Khan")
if input_text:
    # which are chains that execute their links in a predefined order
    # where each step has a single input/output, and the output of one step is the input to the next
    # all the chains are executed sequentially with JSON input/output
    sequential_chain = SequentialChain(
        chains=[llm_chain],
        input_variables=["celebrity_name"])
    response = sequential_chain({'celebrity_name': input_text})

    st.markdown("**Response from OpenAI:**")
    st.write(f"**Full Name**: {response}")

st.markdown("*Powered by OpenAI gpt-3.5-turbo-instruct via LangChain*")
