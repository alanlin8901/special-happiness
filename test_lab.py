from pymilvus import connections, Collection

# 連接 Milvus
connections.connect(alias="default", host="standalone", port="19530")  # 改成你 Milvus 服務的 host 和 port

# 指定你要檢查的 collection 名稱
collection_name = "your_collection_name"

# 載入 collection
collection = Collection(collection_name)

# 取得 collection 資料數量（已插入的實體數）
print("Collection row count:", collection.num_entities)