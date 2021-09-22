from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, pipeline


class Translator:
    def __init__(self, language):
        if language.lower() == "tamil":
            model_name = "Helsinki-NLP/opus-mt-en-dra"
            self.prefix = ">>tam<< "
        elif language.lower() == "hindi":
            model_name = "Helsinki-NLP/opus-mt-en-hi"
            self.prefix = ">>hin<< "
        else:
            raise ValueError(f"{language} is not supported.")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        self.translator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0
        )

    def translate(self, texts, batch_size=32):
        texts = [self.prefix + t for t in texts.tolist()]
        batches = [b for b in self.split_into_batches(texts, batch_size)]
        outputs = []
        with tqdm(total=len(batches), unit="batches") as tbatches:
            for batch in batches:
                outputs += self.translator(
                    batch,
                    clean_up_tokenization_spaces=True,
                    truncation=True
                )
                tbatches.update(1)
        translated_texts = [o["generated_text"] for o in outputs]
        return translated_texts

    def split_into_batches(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]
