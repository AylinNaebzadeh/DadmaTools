from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HF_LanguageModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def score(self, text):
        """Return log probability (base 10) for a given text."""
        encodings = self.tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            encodings = {k: v.to("cuda") for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            # Hugging Face returns loss in nats → convert to log10 probability
            neg_log_likelihood = outputs.loss.item() * encodings["input_ids"].size(1)
            log10_prob = -neg_log_likelihood / torch.log(torch.tensor(10.0)).item()
        return log10_prob

