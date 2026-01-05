import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, deployment_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")
        
        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self._api_key = api_key

    def get_embeddings(
            self, inputs: str | list[str],
            dimensions: int,
            print_response: bool = False
    ) -> dict[int, list[float]]:
        """
        Generate embeddings for input texts using DIAL embeddings API.
        
        Args:
            input_texts: List of text strings to generate embeddings for
            dimensions: Dimension of the embedding vectors (default: 1536)
            
        Returns:
            Dictionary mapping index to embedding vector: {0: [0.1, 0.2, ...], 1: [...], ...}
        """
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"
        }
        request_data = {
            "input": inputs,
            "dimensions": dimensions,
        }

        response = requests.post(url=self._endpoint, headers=headers, json=request_data, timeout=60)

        if response.status_code == 200:
            response_json = response.json()
            data = response_json.get("data", [])
            if print_response:
                print("\n" + "=" * 50 + " RESPONSE " + "=" * 50)
                print(json.dumps(response_json, indent=2))
                print("=" * 108)
            return self._from_data(data)
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    def _from_data(self, data: list[dict]) -> dict[int, list[float]]:
        return {embedding_obj['index']: embedding_obj['embedding'] for embedding_obj in data}


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
