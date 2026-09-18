from dashscope import embeddings
from langchain_core.embeddings import Embeddings
from typing import List
import config_data as config

class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str = config.embeddings_model_name):
        self.model = model
        self.api_key = config.dashscope_api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = embeddings.TextEmbedding.call(
            model=self.model,
            input=texts,
            api_key=self.api_key,
            timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope embedding 调用失败：{resp.code}, {resp.message}")
        return [item["embedding"] for item in resp.output["embeddings"]]

    def embed_query(self, text: str) -> List[float]:
        resp = embeddings.TextEmbedding.call(
            model=self.model,
            input=[text],
            api_key=self.api_key,
            timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope embedding 调用失败：{resp.code}, {resp.message}")
        return resp.output["embeddings"][0]["embedding"]
