import os
import json
import re
import argparse
import pandas as pd
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./v1.0-simplified_simplified-nq-train.jsonl",
        required=False
    )
    parser.add_argument("--quantity", type=int, default=5000, required=False)
    parser.add_argument("--length_limit", type=int, default=40000, required=False)
    parser.add_argument("--output_path", type=str, default="./", required=False)
    return parser.parse_args()


def remove_html(text: str) -> str:
    pattern = re.compile('<.*?>')
    return re.sub(pattern, '', text)


def remove_space(text: str) -> str:
    cleaned_text = re.sub(' +', ' ', text)
    cleaned_text = re.sub(r'\s([?.!",:;](?:\s|$))', r'\1', cleaned_text)
    cleaned_text = cleaned_text.replace(" 's", "'s")
    return cleaned_text.replace(" 't", "'t")


def make_formatted_row(row: Dict, length_limit) -> Dict:
    context = remove_html(row["document_text"])
    context = remove_space(context)
    if len(context) > length_limit:
        context = context[:length_limit]
    question = row["question_text"]
    question = remove_space(question)
    answer_start_token = row["annotations"][0]["short_answers"][0]["start_token"]
    answer_end_token = row["annotations"][0]["short_answers"][0]["end_token"]
    context_tokens = row["document_text"].split(" ")
    answer_tokens = context_tokens[answer_start_token: answer_end_token]
    answer_text = " ".join(answer_tokens)
    answer_text = remove_space(answer_text)
    answer_start = context.find(answer_text)
    return {
        "context": context,
        "question": question,
        "answer_text": answer_text,
        "answer_start": answer_start
    }


if __name__ == "__main__":
    args = parse_args()
    data = []
    with open(args.data_path, "r") as f:
        while len(data) < args.quantity:
            row = json.loads(f.readline())
            if len(row["annotations"]) > 0 and len(row["annotations"][0]["short_answers"]) > 0:
                formatted_row = make_formatted_row(row, args.length_limit)
                if (
                    len(formatted_row["question"]) > 0
                    and len(formatted_row["answer_text"]) > 0
                    and formatted_row["answer_start"] != -1
                ):
                    data.append(formatted_row)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(args.output_path, "natural_questions_en.csv"), index=False)
