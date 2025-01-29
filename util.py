import requests 

import re
import json
import demjson

def extract_json_from_response(response_text):
    return demjson.decode(response_text)
    

def call_llm(prompt):
    URL = "http://localhost:11434/api/generate"
    data = {
  "model": "llama3.2",
  "prompt": prompt,
  "stream": False}
    
    response = requests.post(URL, json=data)
    return response.json()


def add_to_dictionary(originaldict, newValues):
    
    for key, value in newValues.items():
        originaldict[key] = value

    return originaldict


