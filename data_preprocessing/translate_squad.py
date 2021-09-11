import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline

def translate_to_tamil(texts, batch_size=32):
    texts = [">>tam<<" + t for t in texts.tolist()]
    batches = [b for b in split_into_batches(texts, batch_size)]
    outputs = []
    with tqdm(total=len(batches), unit="batches") as tbatches:
        for batch in batches:
            outputs += translator(
                batch, 
                clean_up_tokenization_spaces=True, 
                truncation=True
            )
            tbatches.update(1)
    tamil_texts = [o["generated_text"] for o in outputs]
    return tamil_texts


def split_into_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

        
def find_answer_start(row):
    return row.context.find(row.answer_text)


if __name__ == "__main__":
    model_name = 'Helsinki-NLP/opus-mt-en-dra'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translator = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device=0
    )
    squad_en = pd.read_csv("squad_csv/train-squad.csv")
    tamil_contexts = translate_to_tamil(squad_en.context, batch_size=32)
    tamil_questions = translate_to_tamil(squad_en.question, batch_size=128)
    tamil_texts = translate_to_tamil(squad_en.text, batch_size=128)
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