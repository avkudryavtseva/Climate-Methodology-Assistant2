from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class FlanT5Generator:

    def __init__(self, model_name="google/flan-t5-base"):

        # выбираем устройство - CPU,на GPU падает
        self.device = torch.device("cpu")

        # загружаем tokenizer 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # загружаем FLAN-T5 модель (text-to-text генерация)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # переносим модель на выбранное устройство
        self.model.to(self.device)

        # переводим модель в inference режим (без обучения)
        self.model.eval()

    def generate(self, query: str, context: str) -> str:

        # собираем prompt для модели
        # сюда вставляется вопрос + контекст из retrieval
        prompt = f"""
You are a climate methodology expert.

QUESTION:
{query}

INSTRUCTIONS:
- Answer ONLY using the context
- If not found, say "Not explicitly stated"
- Be concise (3-6 sentences)
- Add citations like VM0012, VM0045 if possible

CONTEXT:
{context}

ANSWER:
"""

        # токенизируем prompt (text -> tokens)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        # переносим входные данные на CPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # отключаем градиенты 
        with torch.no_grad():

            # генерируем ответ модели
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        # декодируем токены обратно в текст
        decoded = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # если модель дублирует "ANSWER:" → чистим вывод
        if "ANSWER:" in decoded:
            decoded = decoded.split("ANSWER:")[-1].strip()

        return decoded
