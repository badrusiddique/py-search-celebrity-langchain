# LangChain Celebrity Search App

This project is a simple web application built with Streamlit and LangChain that allows users to search for information about celebrities. The application demonstrates various features of the LangChain library, from basic LLM usage to more complex sequential chains with memory.

This repository is structured to provide a step-by-step guide to understanding and using LangChain. The files are numbered in a sequence that introduces concepts progressively.

## Getting Started

### Prerequisites

- Python 3.7+
- An OpenAI API key

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/badrusiddique/py-search-celebrity-langchain.git
    cd py-search-celebrity-langchain
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Create a `constants.py` file in the root directory and add your OpenAI API key:
    ```python
    openai_key = "YOUR_OPENAI_API_KEY"
    ```

## Usage

To run the application, use Streamlit. Each numbered file represents a different stage of the application's development.

For example, to run the first version of the app:

```bash
streamlit run 1_using_static.py
```

To run the final version with sequential memory chains:

```bash
streamlit run 7_llm_sequential_memory_chains.py
```

## Project Structure and Evolution

This project is broken down into several files, each demonstrating a new concept in LangChain.

### 1. `1_using_static.py`

This is the most basic version of the application. It demonstrates how to:

-   Initialize the OpenAI LLM from LangChain.
-   Use the LLM to get a response to a static query.
-   Display the response in a Streamlit app.

### 2. `2_using_prompt.py`

This script introduces the concept of `PromptTemplate`. Instead of a static query, we use a template to format the user's input. This allows for more dynamic and structured queries to the LLM.

-   **Key Concept:** `PromptTemplate` for dynamic input.

### 3. `3_llm_chains.py`

Here, we introduce `LLMChain`. A chain is a sequence of calls to components, which can include other chains. In this case, it's a simple chain that takes the user's input, formats it with a `PromptTemplate`, and sends it to the LLM.

-   **Key Concept:** `LLMChain` to combine a prompt and an LLM.

### 4. `4_llm_chains_memory.py`

This script adds memory to the `LLMChain`. The `ConversationBufferMemory` allows the chain to remember previous interactions. This is useful for building conversational applications.

-   **Key Concept:** `ConversationBufferMemory` for stateful chains.

### 5. `5_llm_simple_sequential_chains.py`

This script demonstrates `SimpleSequentialChain`. This type of chain runs multiple chains in sequence, with the output of one chain being the input to the next. It's "simple" because each chain has a single input and a single output.

-   **Key Concepts:** `SimpleSequentialChain` for chaining multiple steps.

### 6. `6_llm_sequential_chains.py`

This script introduces `SequentialChain`, which is a more flexible version of `SimpleSequentialChain`. It allows for multiple inputs and outputs between chains, giving you more control over the data flow.

-   **Key Concepts:** `SequentialChain` for more complex, multi-step workflows.

### 7. `7_llm_sequential_memory_chains.py`

The final script combines `SequentialChain` with memory. Each chain in the sequence gets its own `ConversationBufferMemory`, allowing different parts of the conversation to be stored and recalled independently.

-   **Key Concepts:** `SequentialChain` with independent memory for each chain.

### 8. `llm_fewshot_prompt.py`

This script introduces `FewShotPromptTemplate`. This is a powerful technique where you provide the LLM with a few examples of the desired output format. This helps the model to generate more accurate and consistently formatted responses. The script defines a set of examples and uses them to build a `FewShotPromptTemplate`.

-   **Key Concept:** `FewShotPromptTemplate` for providing in-context examples to the LLM.

## Versions Used

This project uses the following Python packages:

-   `langchain==0.0.202`
-   `streamlit==1.23.1`
-   `openai==0.27.8`

These versions are specified in the `requirements.txt` file.