# Databricks notebook source
# MAGIC %pip install pinecone-client 

# COMMAND ----------

# MAGIC %pip install sentence_transformers

# COMMAND ----------

from sentence_transformers import SentenceTransformer

# We will use embeddings from this model to apply to our data
model = SentenceTransformer(
    "all-mpnet-base-v2", cache_folder= './cache/'
)  # Use a pre-cached model

print('done')

# COMMAND ----------

# load index
from pinecone import Pinecone

PINECONE_API_KEY = ""
pinecone_index_name = "realwise"

pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(pinecone_index_name)

pinecone.describe_index(pinecone_index_name)

# COMMAND ----------

metadata_columns = ['dt_updated', 'high_esr', 'low_esr', 'address', 'type', 'suburb_name', 'postcode', 'state', 'area', 'baths', 'beds', 'cars', 'features', 'price_estimation_price', 'yield_value', 'distance_to_tram', 'closest_tram_stop_name', 'distance_to_train', 'closest_train_stop_name', 'distance_to_rail', 'closest_rail_stop_name', 'distance_to_bus', 'closest_bus_stop_name', 'distance_to_ferry', 'closest_ferry_stop_name', 'distance_to_primary_school', 'closest_primary_school_name', 'distance_to_secondary_school', 'closest_secondary_school_name', 'distance_to_combined_school', 'closest_combined_school_name', 'distance_to_special_school', 'closest_special_school_name', 'nearby_school_names', 'nearby_other_poi']

# COMMAND ----------

example_user_filter = {'type': {'$in': ['House', 'Townhouse', 'Apartment']},
 'suburb_name': {'$in': ['CLAYTON']},
 '$and': [{'beds': {'$gte': 3, '$lte': 3}},
  {'baths': {'$gte': 2, '$lte': 2}},
  {'cars': {'$gte': 1, '$lte': 1}}],
 '$or': [{'$and': [{'low_esr': {'$lte': 900000}}, {'high_esr': {'$gte': 0}}]},
  {'$and': [{'high_esr': {'$eq': 0}},
    {'low_esr': {'$gte': 0}},
    {'low_esr': {'$lte': 900000}}]}]}

# COMMAND ----------

user_filter = {
  'type': {'$in': ['House']},
 #'suburb_name': {'$in': ['CLAYTON','MULGRAVE','HUNGTINGDALE', 'NOTTING HILL', 'GLEN WAVERLEY', 'MOUNT WAVERLEY']},
 '$and': [{'beds': {'$gte': 3, '$lte': 3}}],
 '$or': [{'$and': [{'low_esr': {'$lte': 1500000}}, {'high_esr': {'$gte': 800000}}]},
  {'$and': [{'high_esr': {'$eq': 0}},
    {'low_esr': {'$gte': 0}},
    {'low_esr': {'$lte': 1500000}}]},
    ],
 #'yield_value': {'$gte': 6},
 #'distance_to_rail': {'$lte': 800},
 #'distance_to_primary_school': {'$lte': 1000},
 }

# COMMAND ----------

user_filter = {'type': {'$in': ['House']},
 'suburb_name': {'$in': ['OFFICER']},
 '$and': [{'beds': {'$gte': 2, '$lte': 3}},
  {'area': {'$lte': 600}},
  {'distance_to_rail_upper_bound': {'$lte': 1200}},
  {'distance_to_secondary_school_upper_bound': {'$lte': 1500}},
  {'yield_value': {'$gte': 5.493}}],
 'low_esr': {'$lte': 2000000}}

# COMMAND ----------

user_query = "'modern design. separate space for WFH. granny flat.'"

# COMMAND ----------

user_query_vector = model.encode(user_query).tolist()

pinecone_answer = index.query(
    #namespace="ns1",
    vector=user_query_vector,
    top_k=10,
    include_values=True,
    include_metadata=True,
    filter=user_filter
)



# print some properties
for result in pinecone_answer["matches"]:
    #print(result)
    print(f"score: {result['score']}, property_id: {result['id']}, {result['metadata']['address']}, {result['metadata']['suburb_name']}")
    print("-" * 120)


print("finish")
  
# COMMAND ----------

# res result from summarization
# user_query, user_filter = res_to_query(res)


