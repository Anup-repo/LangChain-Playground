# LangChain-Playground

A comprehensive playground for experimenting with [LangChain](https://github.com/langchain-ai/langchain) chains, models, prompts, and output parsers using various LLM providers (OpenAI, HuggingFace, Google Gemini, Anthropic, etc.). This repository demonstrates practical examples of sequential, parallel, and conditional chains, prompt engineering, structured/unstructured outputs, and embedding models.

## Folder Structure

```
Langchain-chains/
    conditional_chain.py      # Conditional branching chains
    parallel_chain.py         # Parallel execution chains
    sequential_chain.py       # Sequential chains
    simple_chain.py           # Basic chain example

Langchain-models/
    Chatmodels/
        chat_model_anthropic.py   # Anthropic Claude chat model
        chat_model_gemini.py      # Google Gemini chat model
        chat_model_hf_local.py    # Local HuggingFace chat model
        chat_model_hf.py          # HuggingFace Hub chat model
        chat_model_openai.py      # OpenAI chat model
    Embeddingmodels/
        embedding_openai_docs.py  # OpenAI document embeddings
        embedding_openai_query.py # OpenAI query embedding
        embedding_similarity.py   # Embedding similarity example
    LLMs/
        llm_demo.py               # OpenAI LLM demo

Langchain-outputs/
    structured/
        output_json.py            # Structured output with JSON schema
        output_pydantic.py        # Structured output with Pydantic
        output_typed_dict.py      # Structured output with TypedDict
    unstructured/
        json_output_parser.py         # JSON output parser
        pydantic_putput_parser.py     # Pydantic output parser
        str_output_parser.py          # String output parser
        structure_output_parser.py    # Structured output parser

Langchain-prompts/
    chat_history.txt             # Example chat history
    chat_prompt_template.py      # Chat prompt template example
    chatbot.py                   # Simple chatbot with chat history
    dynamic_prompt.py            # Dynamic prompt with user input
    message_place_holder.py      # Message placeholder example
    save_template.py             # Save prompt template to JSON
    static_prompt.py             # Static prompt example
    template.json                # Saved prompt template (JSON)
```

## Features

- **Chains**: Examples of simple, sequential, parallel, and conditional chains.
- **Models**: Integration with OpenAI, HuggingFace, Google Gemini, and Anthropic models.
- **Prompts**: Static and dynamic prompt engineering, chat templates, and message placeholders.
- **Outputs**: Handling structured (Pydantic, TypedDict, JSON Schema) and unstructured outputs.
- **Embeddings**: Document and query embeddings, similarity search.
- **Chatbot**: Simple chatbot with chat history.

## Getting Started

1. **Clone the repository**
    ```sh
    git clone https://github.com/Anup-repo/LangChain-Playground.git
    cd LangChain-Playground
    ```

2. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables**
    - Create `.env` and fill in your API keys for OpenAI, HuggingFace, Google, etc.
    ```sh
    OPENAI_API_KEY=your_openai_api_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    GOOGLE_API_KEY=your_google_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    ```

4. **Run examples**
    - Navigate to any script and run it:
    ```sh
    python Langchain-chains/simple_chain.py
    ```

## Requirements

- Python 3.8+
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://pypi.org/project/openai/)
- [HuggingFace Hub](https://pypi.org/project/huggingface-hub/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [pydantic](https://pypi.org/project/pydantic/)


---

**Note:** This repository is for educational and experimental purposes. Make sure to use your own API keys and comply with the terms of service of each provider.