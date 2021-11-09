import pandas as pd


def process_squad(path, language):
    data = pd.read_csv(path)
    data = data[data.is_in == True]
    data["answer_start"] = data.answers.apply(lambda x: eval(x)[0]["answer_start"])
    data["answer_text"] = data.answers.apply(lambda x: eval(x)[0]["text"])
    data = data[["id", "question", "context", "answer_start", "answer_text"]]
    data["language"] = language
    filename = path.split("/")[-1]
    data.to_csv(filename, index=False)


process_squad("squad_hi.csv", "hindi")
process_squad("squad_ta.csv", "tamil")
