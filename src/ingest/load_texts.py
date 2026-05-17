import os


def load_texts(folder_path):
    """
    Loads all .txt files from folder and returns dict:
    {filename: text}
    """

    data = {}

    for file in os.listdir(folder_path):

        if file.endswith(".txt"):

            path = os.path.join(folder_path, file)

            with open(path, "r", encoding="utf-8") as f:

                data[file] = f.read()

    return data
