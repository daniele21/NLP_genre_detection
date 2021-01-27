import json

def load_json(json_path):
    with open(json_path, 'r') as j:
        json_dict = json.load(j)
        j.close()

    return json_dict