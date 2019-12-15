from nbformat import v3, v4

with open("tweet_model.py") as file_in:
    text = file_in.read()

notebook = v3.reads_py(text)
notebook = v4.upgrade(notebook)

json_form = v4.writes(notebook) + "\n"

with open("tweet_model.ipynb", "w") as file_out:
    file_out.write(json_form)

