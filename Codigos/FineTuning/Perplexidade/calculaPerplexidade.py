import numpy as np
import torch
from transformers import FillMaskPipeline


class PerplexityPipeline(FillMaskPipeline):

    def create_sequential_mask(self, input_data, mask_count=1):
        _, seq_length = input_data["input_ids"].shape
        mask_count = seq_length - 2

        input_ids = input_data["input_ids"]

        new_input_ids = torch.repeat_interleave(input_data["input_ids"], repeats=mask_count, dim=0)
        token_type_ids = torch.repeat_interleave(input_data["token_type_ids"], repeats=mask_count, dim=0)
        attention_mask = torch.repeat_interleave(input_data["attention_mask"], repeats=mask_count, dim=0)
        masked_lm_labels = []
        masked_lm_positions = list(range(1, mask_count + 1))
        for i in masked_lm_positions:
            new_input_ids[i - 1][i] = self.tokenizer.mask_token_id
            masked_lm_labels.append(input_ids[0][i].item())
        new_data = {"input_ids": new_input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
        return new_data, masked_lm_positions, masked_lm_labels

    def __call__(self, input_text, *args, **kwargs):
        """
        Compute perplexity for given sentence.
        """
        if not isinstance(input_text, str):
            return None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1. create sequential mask
        model_inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=100)
        new_data, masked_lm_positions, masked_lm_labels = self.create_sequential_mask(model_inputs)
        model_inputs = new_data
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = torch.tensor(masked_lm_labels).to(device)

        # 2. predict
        self.model = self.model.to(device)
        with torch.no_grad():
            model_outputs = self.model(**model_inputs)

        # 3. compute perplexity
        sentence = {}
        tokens = []
        for i in range(len(labels)):
            model_outputs_i = {}
            model_outputs_i["input_ids"] = model_inputs["input_ids"][i:i + 1]
            model_outputs_i["logits"] = model_outputs["logits"][i:i + 1]
            outputs = self.postprocess(model_outputs_i, target_ids=labels[i:i + 1])
            tokens.append({"token": outputs[0]["token_str"],
                           "prob": outputs[0]["score"]})
        sentence["tokens"] = tokens
        sentence["ppl"] = float(np.exp(- sum(np.log(token["prob"]) for token in tokens) / len(tokens)))
        return sentence
    

def score(model, tokenizer, sentence, device='cuda'):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)

    with torch.no_grad():
        labels = labels.to(device)
        masked_input = masked_input.to(device)
        loss = model(masked_input, labels=labels).loss
    del masked_input, labels
    torch.cuda.empty_cache()
    return np.exp(loss.item())