import json

def save_json(json_dict, filepath):
    with open(filepath, 'w') as j:
        json.dump(json_dict, j, indent=4)
        j.close()

    return