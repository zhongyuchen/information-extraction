import json


length = []
schemas = {}
with open("../data/all_50_schemas", 'r') as f:
  for i, line in enumerate(f):
    spo = json.loads(line)
    s = spo['predicate'] + '的' + spo['subject_type'] + '和' + spo['object_type'] + '是什么？'
    schemas[spo['predicate'] + spo['subject_type']] = s
    if i == 0:
    	print(s)
    length.append(len(s))
print(max(length))
# 20