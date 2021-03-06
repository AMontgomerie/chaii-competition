import argparse
import pandas as pd
from translation import Translator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--questions_only", dest="questions_only", action="store_true")
    parser.add_argument("--qa_batch_size", type=int, default=128, required=False)
    parser.add_argument("--context_batch_size", type=int, default=32, required=False)
    parser.add_argument("--data_path", type=str, default="train-squad.csv", required=False)
    parser.add_argument("--save_path", type=str, default="translated_squad.csv", required=False)
    return parser.parse_args()


def find_answer_start(row):
    return row.context.find(str(row.answer_text))


if __name__ == "__main__":
    config = parse_args()
    translator = Translator(config.language)
    squad_en = pd.read_csv(config.data_path)
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
            squad_en.answer_text,
            batch_size=config.qa_batch_size
        )
    translated_squad = pd.DataFrame({
        "context": squad_en.context if config.questions_only else contexts,
        "question": questions,
        "answer_text": squad_en.answer_text if config.questions_only else texts
    })
    if "id" in squad_en.columns:
        translated_squad["id"] = squad_en.id
    translated_squad["answer_start"] = translated_squad.apply(
        find_answer_start,
        axis=1
    )
    translated_squad["language"] = config.language
    valid_questions = translated_squad.question.str.contains("?", regex=False)
    translated_squad = translated_squad[valid_questions]
    translated_squad.to_csv(config.save_path, index=False)
