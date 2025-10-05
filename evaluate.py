from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset
import evaluate
import torch

def evaluate_model():
    # Load model and tokenizer
    model = BartForConditionalGeneration.from_pretrained("./bart-xsum")
    tokenizer = BartTokenizer.from_pretrained("./bart-xsum")

    # Load test data
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    test_data = dataset["test"]

    # Load metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    predictions = []
    references = []

    for example in test_data:
        input_text = example["article"]
        reference = example["highlights"]

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary
        summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        predictions.append(summary)
        references.append(reference)

    # Compute ROUGE
    rouge_results = rouge.compute(predictions=predictions, references=references)
    print("ROUGE Scores:")
    print(rouge_results)

    # Compute BLEU
    bleu_results = bleu.compute(predictions=predictions, references=references)
    print("BLEU Score:")
    print(bleu_results)

if __name__ == "__main__":
    evaluate_model()
