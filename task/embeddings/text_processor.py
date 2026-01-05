from enum import StrEnum

import pg8000

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
        return pg8000.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def _truncate_table(self):
        """Truncate the vectors table"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors")
                conn.commit()
                print("Table has been successfully truncated.")

    def _save_chunk(self, embedding: list[float], chunk: str, document_name: str):
        """Save chunk with embedding to database"""
        vector_string = f"[{','.join(map(str, embedding))}]"

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                    (document_name, chunk, vector_string)
                )
                conn.commit()

        print(f"Stored chunk from document: {document_name}")

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
            file_name: str,
            chunk_size: int,
            overlap: int,
            dimensions: int,
            truncate_table: bool = True,
    ):
        """
        Load content from file, chunk it, generate embeddings, and save to DB
        Args:
            file_name: path to file
            chunk_size: chunk size (min 10 chars)
            overlap: overlap chars between chunks
            dimensions: number of dimensions to store
            truncate_table: truncate table if true
        """
        if chunk_size < 10:
            raise ValueError("chunk_size must be at least 10")
        if overlap < 0:
            raise ValueError("overlap must be at least 0")
        if overlap >= chunk_size:
            raise ValueError("overlap should be lower than chunkSize")

        if truncate_table:
            self._truncate_table()

        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks: list[str] = chunk_text(content, chunk_size, overlap)
        embeddings: dict[int, list[float]] = self.embeddings_client.get_embeddings(chunks, dimensions)

        print(f"Processing document: {file_name}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total embeddings: {len(embeddings)}")

        for i in range(len(chunks)):
            self._save_chunk(embeddings.get(i), chunks[i], file_name)


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
            search_mode: SearchMode,
            user_request: str,
            top_k: int,
            score_threshold: float,
            dimensions: int
    ) -> list[str]:
        """
        Perform similarity search
        Args:
            search_mode: Search mode (Cosine or Euclidian distance)
            user_request: User request
            top_k: Number of results to return
            score_threshold: Minimum score to return (range 0.0 -> 1.0)
            dimensions: Number of dimensions to return (has to be the same as data persisted in VectorDB)
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if score_threshold < 0 or score_threshold > 1:
            raise ValueError("score_threshold must be in [0.0..., 0.99...] range")

        query_embedding = self.embeddings_client.get_embeddings(inputs=user_request, dimensions=dimensions)[0]
        vector_string = f"[{','.join(map(str, query_embedding))}]"

        if search_mode == SearchMode.COSINE_DISTANCE:
            max_distance = 1.0 - score_threshold
        else:
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        retrieved_chunks = []
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self._get_search_query(search_mode), (vector_string, vector_string, max_distance, top_k))
                results = cursor.fetchall()

                for row in results:
                    # pg8000 returns tuples: (text, distance)
                    text, distance = row[0], row[1]
                    if search_mode == SearchMode.COSINE_DISTANCE:
                        similarity = 1.0 - distance
                    else:
                        similarity = 1.0 / (1.0 + distance)

                    print(f"---Similarity score: {similarity:.2f}---")
                    print(f"Data: {text}\n")
                    retrieved_chunks.append(text)

        return retrieved_chunks

    def _get_search_query(self, search_mode: SearchMode) -> str:
        return """SELECT text, embedding {mode} %s::vector AS distance
                  FROM vectors
                  WHERE embedding {mode} %s::vector <= %s
                  ORDER BY distance
                  LIMIT %s""".format(mode='<->' if search_mode == SearchMode.EUCLIDIAN_DISTANCE else '<=>')

