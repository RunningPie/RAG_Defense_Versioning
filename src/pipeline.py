import os
import yaml
from .rag_components.retriever import Retriever
from .rag_components.generator import Generator

class RAGPipeline:
    """
    Orchestrates the entire Retrieval-Augmented Generation (RAG) pipeline,
    combining the retriever and the generator.
    """
    def __init__(self, config_path='config.yaml'):
        """
        Initializes the RAG pipeline by loading configuration and setting up
        the retriever and generator components.

        Args:
            config_path (str): Path to the main configuration file.
        """
        print("Initializing RAG Pipeline...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.retriever = Retriever(self.config)
        self.generator = Generator(self.config)
        
        # Build the retriever's index immediately upon initialization
        kb_path = os.path.join(
            self.config['data_paths']['processed_data_dir'],
            self.config['filenames']['clean_kb']
        )
        self.retriever.build_index(kb_path)

    async def run(self, query: str, generator_model: str) -> str:
        """
        Executes a query through the full RAG pipeline.

        Args:
            query (str): The user's input query.
            generator_model (str): The name of the generator LLM to use.

        Returns:
            str: The final generated answer.
        """
        print(f"\n--- Running pipeline for query: '{query}' ---")
        
        # 1. Retrieve relevant context
        print("Step 1: Retrieving context...")
        context = self.retriever.search(query)
        if not context:
            return "Could not find any relevant context for your query."
        
        print(f"Retrieved {len(context)} documents.")
        
        # 2. Format the prompt
        print("Step 2: Formatting prompt for the generator...")
        messages = self.generator.format_prompt(query, context)
        
        # 3. Generate a response
        print(f"Step 3: Generating response with model '{generator_model}'...")
        answer = await self.generator.generate_response(generator_model, messages)
        
        print("--- Pipeline run complete. ---")
        return answer

# Example usage (can be placed in a main.py or a test script)
async def example_run():
    pipeline = RAGPipeline(config_path='config.yaml')
    
    # Select one of the models from your config
    model_to_test = pipeline.config['generator_models'][0]
    
    # Example query
    query = "Are there any movies about toys that come to life?"
    
    response = await pipeline.run(query, model_to_test)
    
    print("\n\n--- FINAL RESPONSE ---")
    print(response)

# To run this example, you would need an async entry point
if __name__ == '__main__':
    import asyncio
    asyncio.run(example_run())
