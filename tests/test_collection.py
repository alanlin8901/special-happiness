from pymilvus import utility, connections, Collection
# 連接 Milvus（記得改成你的 Milvus 地址與埠）
connections.connect("default", host="localhost", port="19531")
collection_name = "lab_papers"
collection = Collection(collection_name)

print(f"Collection '{collection_name}' contains {collection.num_entities} entities.")