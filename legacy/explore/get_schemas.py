import json


schemas = {}
with open("all_50_schemas", 'r') as f:
  for i, line in enumerate(f):
    spo = json.loads(line)
    schemas[spo['predicate'] + spo['subject_type']] = i
print(schemas)
