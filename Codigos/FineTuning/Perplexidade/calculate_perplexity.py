from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
from datasets import load_dataset
import os
import math
import json
import warnings
os.environ["WANDB_DISABLED"] = "true"
warnings.filterwarnings("ignore")


def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, max_length=128)


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        num_proc=4
    )


def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


def evaluate_model(model_path, dataset, batch_size):
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))
    
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    data_collator = get_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        save_total_limit=1,
        per_device_eval_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results['eval_loss'])
    return perplexity


def evaluate_multiple_models(models, dataset, batch_size, dir_save_metrics):
    results = {}
    for modelo in models:
        print(f"Avaliando o modelo: {modelo.get('model_name')}")
        perplexity = evaluate_model(modelo.get('modelo'), dataset, batch_size)
        results[modelo.get('model_name')] = perplexity
        
    return results


def main() -> None:
    params = json.load(open("configPerplexidade.json"))
    dataset = load_dataset("csv", data_files=params['dataset'])
    perplexity_results = evaluate_multiple_models(params['modelos'], dataset['train'], params['batch_size'], params['dir_save_metrics'])
    json.dump(perplexity_results, open(os.path.join(params['dir_save_metrics'], f"{params['name_metrics']}.json"), "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()