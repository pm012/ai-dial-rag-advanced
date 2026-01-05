from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a RAG-powered assistant specialized in helping users with microwave oven operations and maintenance.

Your responses are based on the provided RAG Context from the microwave manual. You will receive messages in the following structure:
1. RAG Context: Relevant excerpts from the microwave manual
2. User Question: The actual question from the user

Instructions:
- Use ONLY the information from the RAG Context to answer questions
- Focus exclusively on microwave-related topics (operation, safety, cleaning, cooking, maintenance)
- If the question is not related to microwave usage or cannot be answered using the provided context, politely decline and explain that you can only assist with microwave-related questions based on the manual
- Do not answer questions outside the scope of microwave operations, even if you have general knowledge about them
- Be concise, accurate, and helpful in your responses
"""

# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """RAG Context: {context}

User Question: {question}"""


# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)

def main():
    # Initialize clients and processor
    embeddings_client = DialEmbeddingsClient('text-embedding-3-small-1', API_KEY)
    chat_client = DialChatCompletionClient('gpt-4o', API_KEY)
    
    db_config = {
        'host': 'localhost',
        'port': 5433,
        'database': 'vectordb',
        'user': 'postgres',
        'password': 'postgres'
    }
    
    text_processor = TextProcessor(embeddings_client, db_config)
    
    # Process the microwave manual (only needs to be done once)
    print("Processing microwave manual...")
    text_processor.process_text_file(
        'task/embeddings/microwave_manual.txt',
        chunk_size=300,
        overlap=40,
        truncate_table=True
    )
    print("Manual processed and stored in vector database.\n")
    
    # Initialize conversation
    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))
    
    print("RAG Microwave Assistant")
    print("=" * 50)
    print("Ask questions about microwave operation and maintenance.")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    # Main chat loop
    while True:
        # Get user input from console
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        # Retrieve context
        print("\n[RETRIEVAL] Searching for relevant context...")
        context_chunks = text_processor.search(
            user_input,
            mode=SearchMode.COSINE_DISTANCE,
            top_k=5,
            min_score=0.5
        )
        
        print(f"[RETRIEVAL] Found {len(context_chunks)} relevant chunks")
        
        # Perform augmentation
        context_text = "\n\n".join(context_chunks)
        augmented_prompt = USER_PROMPT.format(context=context_text, question=user_input)
        
        print(f"[AUGMENTATION] Prompt created (length: {len(augmented_prompt)} chars)")
        
        # Add user message to conversation
        conversation.add_message(Message(Role.USER, augmented_prompt))
        
        # Perform generation
        print("[GENERATION] Generating response...\n")
        response = chat_client.get_completion(conversation.get_messages())
        
        # Add AI response to conversation
        conversation.add_message(response)
        
        # Display response
        print(f"Assistant: {response.content}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    # PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
    # RUN docker-compose.yml
    main()