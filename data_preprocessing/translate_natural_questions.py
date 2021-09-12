import pandas as pd
from translation import EnTaTranslator

if __name__ == "__main__":
    translator = EnTaTranslator()
    data = pd.read_csv("chaii-competition/extra_data/natural_questions_en.csv")
    data["question"] = translator.translate_to_tamil(data.question, batch_size=256)
    data.to_csv("chaii-competition/extra_data/natural_questions_ta.csv", index=False)