import os
from groq import AsyncGroq

class Generator:
    """
    Handles interactions with the Groq API to generate responses from an LLM.
    """
    def __init__(self, config: dict):
        """
        Initializes the Generator with API configuration.

        Args:
            config (dict): The project configuration dictionary.
        """
        print("Initializing Generator...")
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Please set it as an env variable or in config.yaml.")
        
        self.client = AsyncGroq(api_key=groq_api_key)

    def format_prompt(self, query: str, context: list[dict]) -> list[dict]:
        """
        Formats the prompt for the LLM with the retrieved context.

        Args:
            query (str): The user's original query.
            context (list[dict]): A list of context documents from the retriever.

        Returns:
            list[dict]: The formatted list of messages for the Groq API.
        """
        context_str = "\n\n---\n\n".join([
            f"Title: {item['document']['title']}\nDescription: {item['document']['description']}"
            for item in context
        ])

        system_prompt = (
            "You are a helpful movie recommender assistant. Answer the user's question based *only* "
            "on the provided context about movie descriptions. If the context does not contain the "
            "answer, state that you cannot find the information in the provided documents."
        )
        
        user_prompt = (
            f"Here is the context I found:\n\n"
            f"{context_str}\n\n"
            f"Based on this context, please answer the following question: {query}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    async def generate_response(self, model_name: str, messages: list[dict]) -> str:
        """
        Generates a response from the specified Groq model.

        Args:
            model_name (str): The name of the model to use (e.g., 'gemma2-9b-it').
            messages (list[dict]): The list of messages forming the prompt.

        Returns:
            str: The content of the generated response.
        """
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.5, # Lower temperature for more factual responses
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while calling the Groq API: {e}")
            return "Error: Could not get a response from the language model."

