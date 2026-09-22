import config_data as config
from langchain_milvus import Milvus


class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding

        # Milvus向量库初始化
        self.vector_store = Milvus(
            embedding_function=self.embedding,
            collection_name=config.collection_name,
            # Milvus连接参数：本地单机 / Zilliz云端都在这里配置
            connection_args={
                "uri": config.milvus_uri,       # 本地示例: "http://localhost:19530"
                # "token": config.milvus_token, # 本地Milvus不需要token；Zilliz云才需要
            },
            auto_id=True,  # 自动生成主键ID
        )

    def get_retriever(self):
        # 检索top-k，和原来Chroma接口保持一致
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})


if __name__ == '__main__':
    from MyDashScopeEmbeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我体重是180斤，推荐一个尺码")
    print(res)
