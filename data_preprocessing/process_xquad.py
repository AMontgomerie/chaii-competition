import argparse
import pandas as pd
import json
from glob import glob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./", required=False)
    parser.add_argument("--output_path", type=str, default="./xquad.csv", required=False)

    return parser.parse_args()


def process_xquad(data_dir):
    data_files = glob(f"{data_dir}/**/*.json", recursive=True)
    df_list = []

    for data_file in data_files:
        language = data_file.split(".")[-2]
        df = process_data(data_file, language)
        df_list.append(df)

    return pd.concat(df_list)

 
def process_data(path, language):
    raw_df = pd.read_json(path)
    processed_data = []
    
    for i, row in raw_df.iterrows():
        data = row["data"]
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                processed_row = {
                    "context": context,
                    "question": qa["question"],
                    "answer_start": qa["answers"][0]["answer_start"],
                    "answer_text": qa["answers"][0]["text"],
                    "language": language
                }
                processed_data.append(processed_row)

    return pd.DataFrame(processed_data)

if __name__ == "__main__":
    args = parse_args()
    data = process_xquad(args.data_path)
    data.to_csv(args.output_path, index=False)