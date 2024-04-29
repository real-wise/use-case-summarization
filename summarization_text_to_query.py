# Databricks notebook source
# MAGIC %pip install openai

# COMMAND ----------

import os
from openai import OpenAI
import json
import re

#API Keys
os.environ['OPENAI_API_KEY'] = ''

# COMMAND ----------


json_schema = {
  "summary":
  "summary the text without all the numeric features",
  "property_type": "must be in one of 'House', 'Townhouse', 'Apartment'",  # to add more
  "suburb_name":
  "a list. location information, must be an Australian suburb name",
  "bathroom_lower_bound":
  "number of bathrooms minimum",
  "bathroom_upper_bound":
  "number of bathrooms maximum",
  "bedroom_lower_bound":
  "number of bedrooms minimum",
  "bedroom_upper_bound":
  "number of bedrooms maximum",
  "carpark_lower_bound":
  "number of car parks minimum",
  "carpark_upper_bound":
  "number of car parks maximum",
  "area_lower_bound":
  "minimum area in squared meters. must be in integer format",
  "area_upper_bound":
  "maximum area in squared meters. must be in integer format",
  "budget_lower_bound":
  "minimum budget. must be in integer format",
  "budget_upper_bound":
  "maximum budget. must be in integer format",
  "distance_to_rail_upper_bound":
  "distance to rail station. must be in integer format",
  "distance_to_primary_school_upper_bound":
  "distance to primary school. must be in integer format",
  "distance_to_secondary_school_upper_bound":
  "distance to secondary school. must be in integer format",
  "yield_value_lower_bound":
  "minimum value of annual rental yield. value greater than 0.",
  "yield_value_upper_bound":
  "maximum value of annual rental yield. value greater than 0."
}

json_schema_str = ', '.join([f"'{key}': {value}" for key, value in json_schema.items()])



# COMMAND ----------

system_prompt = f"You are a friendly assistant to summarise the property search preference and spliting out numerics and non-numerics values. If you do not know the information, do not assume and do not fill in the information. The JSON schema should include {json_schema_str}."

# COMMAND ----------

example1 = ' Example text 1: I want to buy a house with a mountain view. It must have air conditioning and separated dining room. It has to be north or east facing. Near the supermarkets. Close to hospital. single storey. Land size greater than 600m2. It locates in Ferntree Gully. price of 600k$ to 1000000$. Example output 1: {"summary": "mountain view. air conditioning and separated dining room. north or east facing. near supermarkets and hospital. single storey.", "suburb_name": ["Ferntree Gully"], "property_type": ["House"], "area_lower_bound": 600, budget_lower_bound": 600000, "budget_upper_bound": 1000000}'

# COMMAND ----------

example2 = ' Example text 2: looking for a townhouse or apartment with ocean view. near the seaside, in the following suburb: mornington, seaford, dromana. less than 10 minutes drive to shops. big mansion. Walking distance to rail station(less than 1000m). 2-4 bedrooms, I have a budget between 1.5M and 3M. must have at least 3 car garages. total size greater than 600. Example output 2: {"summary": "ocean view. near the seaside. less than 10 minutes drive to shops. big mansion.", "suburb_name": ["Mornington", "Seaford", "Dromana"], "property_type": ["Townhouse", "Apartment"],"bedroom_lower_bound": 2, "bedroom_upper_bound": 4, "carpark_lower_bound": 3, "area_lower_bound":600, "budget_lower_bound": 1500000, "budget_upper_bound": 3000000, "distance_to_rail_upper_bound": 1000}'

# COMMAND ----------

example3 = ' Example text 3: a place has modern design. strong rental yield of more than 6% rental return. 2 bedrooms. Example output 3: {"summary": "modern design.","bedroom_lower_bound": 2, "bedroom_upper_bound": 2, "yield_value_lower_bound": 6}'

# COMMAND ----------

example4 = " Example text 4: I want to buy in a place with strong rental income at least 7.523% return, nearby clayton, below 900k price, 3 bedroom, 2 bath, at least 1 car space, modern style, less than 5 year old. land size greater than 500m2. Example output 4: {'summary': 'strong rental income. modern style.', 'suburb_name': ['Clayton'], 'property_type': ['House'], 'budget_upper_bound': 900000, 'bedroom_lower_bound': 3, 'bedroom_upper_bound': 3, 'bathroom_lower_bound': 2, 'bathroom_upper_bound': 2, 'carpark_lower_bound': 1, 'carpark_upper_bound': 1, 'area_lower_bound': 500, 'yield_value_lower_bound': 7.523}"

# COMMAND ----------

example5 = " Example text 5: I want to buy a place near pakenham. modern design. less than 800m to train station. 1-2 bedrooms, budget less than 1M$. garage can park at least 2 cars. land size > 600. less than 1000m to local primary school. rental return is greater than four percent per annum  Example output 5: {'summary': 'modern design. close to train station. close to local primary school.', 'suburb_name': ['Pakenham'], 'property_type': ['House', 'Townhouse', 'Apartment'], 'bedroom_lower_bound': 1, 'bedroom_upper_bound': 2, 'carpark_lower_bound': 2, 'area_lower_bound': 600, 'budget_upper_bound': 1000000, 'distance_to_rail_upper_bound': 800, 'distance_to_primary_school_upper_bound': 1000, 'yield_value_lower_bound': 4}"

# COMMAND ----------

# chinese mum and dad want to send kids to good school and live in chinese suburb
#text = 'I want to buy a house in a good school zone. we need 3 bedrooms for kids and family.2 seperate bathrooms for kids and us. it needs to be 1.5 - 2M$.'

# city boy
#text = 'I want to buy a house in Officer. modern design. less than 1200m to train station. 2-3 bedrooms, budget less than 2M$. land size < 600. less than 1500m to local secondary school. rental return is greater than 5.493'

#old man
text = 'a giant house with country view, victorian style, not too far away from dandenong. at least 4 bedrooms and 2 bathrooms. i have a budget of more than 2.5M$'

#text = "I want to buy in a place with strong rental income at least 5% return, nearby clayton, below 900k price, 3 bedroom, 2 bath, at least 1 car space, modern style, less than 5 year old. land size greater than 500m2."

# COMMAND ----------

#col("sp.rail")[0]["distance"].alias("distance_to_rail"),
#col("sp.primary_school")[0]["distance"].alias("distance_to_primary_school"),
#col("sp.secondary_school")[0]["distance"].alias("distance_to_secondary_school"),
#nearby_school_names
#yield_value
#nearby_other_poi bank, hospital clinic medical centre, supermarket, (foodwork coles woolworths aldi) park beach

# COMMAND ----------


# try
# need to force schema if possible.

system_prompt_with_few_shots=system_prompt + example1 + example2 + example3 + example4 + example5




from openai import OpenAI


client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)


response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": system_prompt_with_few_shots},
    {"role": "user", "content": text}
  ]
)

res = response.choices[0].message.content

#print(res)

# COMMAND ----------

import json
# Parse JSON string to a JSON object
res_json = json.loads(res)

#print(res_json)

def res_to_query(input_data):
  output = {}

  if 'property_type' in input_data:
    output["type"] = {"$in": input_data['property_type']}

  if 'suburb_name' in input_data:
    output["suburb_name"] = {
        "$in":
        [suburb_name.upper() for suburb_name in input_data['suburb_name']]
    }

  if 'bedroom_lower_bound' in input_data or 'bedroom_upper_bound' in input_data:
    bedroom_query = {}
    if 'bedroom_lower_bound' in input_data:
      bedroom_query["$gte"] = input_data['bedroom_lower_bound']
    if 'bedroom_upper_bound' in input_data:
      bedroom_query["$lte"] = input_data['bedroom_upper_bound']
    output["$and"] = output.get("$and", []) + [{"beds": bedroom_query}]

  if 'bathroom_lower_bound' in input_data or 'bathroom_upper_bound' in input_data:
    bathroom_query = {}
    if 'bathroom_lower_bound' in input_data:
      bathroom_query["$gte"] = input_data['bathroom_lower_bound']
    if 'bathroom_upper_bound' in input_data:
      bathroom_query["$lte"] = input_data['bathroom_upper_bound']
    output["$and"] = output.get("$and", []) + [{"baths": bathroom_query}]

  if 'carpark_lower_bound' in input_data or 'carpark_upper_bound' in input_data:
    carpark_query = {}
    if 'carpark_lower_bound' in input_data:
      carpark_query["$gte"] = input_data['carpark_lower_bound']
    if 'carpark_upper_bound' in input_data:
      carpark_query["$lte"] = input_data['carpark_upper_bound']
    output["$and"] = output.get("$and", []) + [{"cars": carpark_query}]

  if 'area_lower_bound' in input_data or 'area_upper_bound' in input_data:
    area_query = {}
    if 'area_lower_bound' in input_data:
      area_query["$gte"] = input_data['area_lower_bound']
    if 'area_upper_bound' in input_data:
      area_query["$lte"] = input_data['area_upper_bound']
    output["$and"] = output.get("$and", []) + [{"area": area_query}]

  if ('budget_lower_bound' in input_data) and ('budget_upper_bound'
                                               in input_data):
    #(low_esr < upper bound and high_esr > lower_bound) or (high_esr=0 and low_esr>lower_bound and low_esr<upper_bound)
    # allow 0 to include properties with only low_esr

    or_condition_1 = {
        "$and": [{
            "low_esr": {
                "$lte": input_data['budget_upper_bound']
            }
        }, {
            "high_esr": {
                "$gte": input_data['budget_lower_bound']
            }
        }]
    }
    or_condition_2 = {
        "$and": [{
            "high_esr": {
                "$eq": 0
            }
        }, {
            "low_esr": {
                "$gte": input_data['budget_lower_bound']
            }
        }, {
            "low_esr": {
                "$lte": input_data['budget_upper_bound']
            }
        }]
    }
    output["$or"] = output.get("$or", []) + [or_condition_1, or_condition_2]

  if ('budget_lower_bound' in input_data) and ('budget_upper_bound'
                                               not in input_data):
    #(high_esr == 0 and low_esr>budget_lower_bound) or (high_esr>budget_lower_bound)
    output["$or"] = output.get("$or", []) + [{
        "high_esr": {
            "$gte": input_data['budget_lower_bound']
        }
    }, {
        "$and": [{
            "high_esr": {
                "$eq": 0
            }
        }, {
            "low_esr": {
                "$gte": input_data['budget_lower_bound']
            }
        }]
    }]

  if ('budget_lower_bound' not in input_data) and ('budget_upper_bound'
                                                   in input_data):
    output["low_esr"] = {"$lte": input_data['budget_upper_bound']}

  distance_query = {}
  if 'distance_to_rail_upper_bound' in input_data:
    distance_query["$lte"] = input_data['distance_to_rail_upper_bound']
    output["$and"] = output.get("$and", []) + [{
        "distance_to_rail_upper_bound":
        distance_query
    }]

  distance_query = {}
  if 'distance_to_primary_school_upper_bound' in input_data:
    distance_query["$lte"] = input_data[
        'distance_to_primary_school_upper_bound']
    output["$and"] = output.get("$and", []) + [{
        "distance_to_primary_school_upper_bound":
        distance_query
    }]

  distance_query = {}
  if 'distance_to_secondary_school_upper_bound' in input_data:
    distance_query["$lte"] = input_data[
        'distance_to_secondary_school_upper_bound']
    output["$and"] = output.get("$and", []) + [{
        "distance_to_secondary_school_upper_bound":
        distance_query
    }]

  if 'yield_value_lower_bound' in input_data or 'yield_value_upper_bound' in input_data:
    yield_query = {}
    if 'yield_value_lower_bound' in input_data:
      yield_query["$gte"] = input_data['yield_value_lower_bound']
    if 'yield_value_upper_bound' in input_data:
      yield_query["$lte"] = input_data['yield_value_upper_bound']
    output["$and"] = output.get("$and", []) + [{"yield_value": yield_query}]

  return (input_data['summary'], output)


# COMMAND ----------

user_query, user_filter = res_to_query(res_json)

# COMMAND ----------

#res

# COMMAND ----------

print('user_query:\n', user_query)

# COMMAND ----------

print('user_filter:\n', user_filter)

# COMMAND ----------



