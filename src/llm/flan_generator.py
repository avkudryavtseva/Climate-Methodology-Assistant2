from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class FlanT5Generator:

    def __init__(self, model_name="google/flan-t5-base"):

        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    # ✅ ВОТ ТУТ ОТСТУП (КРИТИЧНО!)
    def generate(self, query: str, context: str) -> str:

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

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "ANSWER:" in decoded:
            decoded = decoded.split("ANSWER:")[-1].strip()

        return decoded
