import pandas as pd
from translation import EnTaTranslator

def find_answer_start(row):
    return row.context.find(row.answer_text)


if __name__ == "__main__":
    translator = EnTaTranslator()
    squad_en = pd.read_csv("squad_csv/train-squad.csv")
    tamil_contexts = translator.translate_to_tamil(squad_en.context, batch_size=32)
    tamil_questions = translator.translate_to_tamil(squad_en.question, batch_size=128)
    tamil_texts = translator.translate_to_tamil(squad_en.text, batch_size=128)
    tamil_squad = pd.DataFrame({
        "id": squad_en.id,
        "context": tamil_contexts,
        "question": tamil_questions,
        "answer_text": tamil_texts
    })
    tamil_squad["answer_start"] = tamil_squad.apply(find_answer_start, axis=1)
    tamil_squad["language"] = ["tamil"]*len(squad_en)
    tamil_squad.to_csv("tamil_squad.csv", index=False)
    print(tamil_squad.sample(10))