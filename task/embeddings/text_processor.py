from enum import StrEnum

import psycopg

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        """Truncate the vectors table"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE vectors;")
                conn.commit()
        finally:
            conn.close()

    def _save_chunk(self, document_name: str, text: str, embedding: list[float]):
        """Save a text chunk with its embedding to the database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Convert embedding to string representation and cast to vector type
                embedding_str = str(embedding)
                cur.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (document_name, text, embedding_str)
                )
                conn.commit()
        finally:
            conn.close()

    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)
    
    def process_text_file(
        self, 
        file_path: str, 
        chunk_size: int = 300, 
        overlap: int = 40, 
        dimensions: int = 1536, 
        truncate_table: bool = False
    ):
        """
        Process a text file: load, chunk, embed, and store in database.
        
        Args:
            file_path: Path to the text file to process
            chunk_size: Size of text chunks (default: 300)
            overlap: Character overlap between chunks (default: 40)
            dimensions: Embedding dimensions (default: 1536)
            truncate_table: Whether to truncate the vectors table before inserting (default: False)
        """
        # Truncate table if needed
        if truncate_table:
            print("Truncating vectors table...")
            self._truncate_table()
        
        # Load content from file
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # Generate chunks
        chunks = chunk_text(text_content, chunk_size, overlap)
        print(f"Generated {len(chunks)} chunks from the document")
        
        # Generate embeddings from chunks
        embeddings_dict = self.embeddings_client.get_embeddings(chunks, dimensions=dimensions)
        
        # Get document name from file path
        document_name = file_path.split('/')[-1]
        
        # Save embeddings and chunks to DB
        for idx, chunk in enumerate(chunks):
            embedding = embeddings_dict[idx]
            self._save_chunk(document_name, chunk, embedding)
        
        print(f"Successfully saved {len(chunks)} chunks to database")


    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`
    
    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.COSINE_DISTANCE,
        top_k: int = 5,
        min_score: float = 0.5,
        dimensions: int = 1536
    ) -> list[str]:
        """
        Search for relevant text chunks using vector similarity.
        
        Args:
            query: Search query text
            mode: Distance metric to use (cosine or euclidean)
            top_k: Number of top results to return (default: 5)
            min_score: Minimum similarity score threshold (default: 0.5)
            dimensions: Embedding dimensions (default: 1536)
            
        Returns:
            List of relevant text chunks
        """
        # Generate embeddings from user request
        embeddings_dict = self.embeddings_client.get_embeddings([query], dimensions=dimensions)
        query_embedding = embeddings_dict[0]
        
        # Determine distance operator based on search mode
        distance_operator = "<=>" if mode == SearchMode.COSINE_DISTANCE else "<->"
        
        # Search in DB relevant context
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Convert embedding to string and cast to vector
                embedding_str = str(query_embedding)
                
                # Create select query with distance calculation
                query_sql = f"""
                    SELECT text, embedding {distance_operator} %s::vector as distance
                    FROM vectors
                    WHERE embedding {distance_operator} %s::vector < %s
                    ORDER BY distance
                    LIMIT %s
                """
                
                cur.execute(query_sql, (embedding_str, embedding_str, min_score, top_k))
                results = cur.fetchall()
                
                # Extract text from results
                texts = [row[0] for row in results]
                return texts
        finally:
            conn.close()

