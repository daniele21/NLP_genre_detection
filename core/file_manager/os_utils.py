import os

def exists(path):
    return os.path.exists(path)


def ensure_folder(folder):
    if(exists(folder) == False):
        os.makedirs(folder)
        return True

    return False