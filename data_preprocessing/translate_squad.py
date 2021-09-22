import argparse
import pandas as pd
from translation import Translator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--questions_only", dest="questions_only", action="store_true")
    parser.add_argument("--qa_batch_size", type=int, default=128, required=False)
    parser.add_argument("--context_batch_size", type=int, default=32, required=False)
    return parser.parse_args()


def find_answer_start(row):
    return row.context.find(row.answer_text)


if __name__ == "__main__":
    config = parse_args()
    translator = Translator(config.language)
    squad_en = pd.read_csv("squad_csv/train-squad.csv")
    questions = translator.translate(
        squad_en.question,
        batch_size=config.qa_batch_size
    )
    if not config.questions_only:
        contexts = translator.translate(
            squad_en.context,
            batch_size=config.context_batch_size
        )
        texts = translator.translate(
            squad_en.text,
            batch_size=config.qa_batch_size
        )
    translated_squad = pd.DataFrame({
        "id": squad_en.id,
        "context": squad_en.contexts if config.questions_only else contexts,
        "question": questions,
        "answer_text": squad_en.text if config.questions_only else texts
    })
    translated_squad["answer_start"] = translated_squad.apply(
        find_answer_start,
        axis=1
    )
    translated_squad["language"] = [config.language]*len(squad_en)
    translated_squad.to_csv(f"{config.language}_squad.csv", index=False)
    print(translated_squad.sample(10))
