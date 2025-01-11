from asyncio import run

from src.core.mountains_ner import MountainsNER


async def main() -> None:
    return await MountainsNER().running()


if __name__ == '__main__':
    run(main())

# import torch
# from transformers import BertForTokenClassification, BertTokenizer, AutoTokenizer, AutoModelForTokenClassification
#
# # Load the model and tokenizer
# model = AutoModelForTokenClassification.from_pretrained('Gepe55o/mountain-ner-bert-base')
# """
# Model Description: mountain-ner-bert-base is a fine-tuned model based on the BERT base architecture for mountain names Entity Recognition tasks.
# """
# tokenizer = AutoTokenizer.from_pretrained('Gepe55o/mountain-ner-bert-base')
#
# model.to(torch.device('cpu'))
# text ='Mount Everest is the highest mountain in the world.'
# inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
# with torch.no_grad():
#     outputs = model(**inputs)
# logits = outputs.logits
# predicted_class_ids = torch.argmax(logits, dim=2)
#
# # tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
# tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
# predictions = predicted_class_ids.squeeze().tolist()
#
# for token, prediction in zip(tokens, predictions):
#     print(f"Token: {token}, Prediction: {prediction}")

