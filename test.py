from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch


tokenizer = GPT2Tokenizer.from_pretrained('microsoft/dialogrpt')
model = GPT2ForSequenceClassification.from_pretrained('microsoft/dialogrpt')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits