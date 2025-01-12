from collections import Counter

import pandas as pd

from loguru import logger

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset


from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt


class MountainsNER:

    @staticmethod
    async def get_dataset():
        """
        Here I import and use preprocessed dataset (telord/mountains-ner-dataset).
        """
        dataset = load_dataset("telord/mountains-ner-dataset")
        logger.info(dataset)
        return dataset

    class MountainNERDataset(Dataset):
        """
        Dataset class to handle token classification data.
        """
        def __init__(self, dataset, tokenizer, max_length=512):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sentence = self.dataset[idx]['sentence']
            labels = self.dataset[idx]['labels']
            encoding = self.tokenizer(
                [sentence],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt',
                is_split_into_words=False
            )

            # Get the input_ids, attention_mask, and labels
            input_ids = encoding['input_ids'].squeeze(0)  # Remove the batch dimension for single input
            attention_mask = encoding['attention_mask'].squeeze(0)

            # Pad labels to match the input sequence length
            labels = labels + [0] * (self.max_length - len(labels))  # Pad labels to match max_length

            # Create the tensor for labels
            encoding['labels'] = torch.tensor(labels, dtype=torch.long)

            # Get the input_ids and attention_mask from the encoding
            input_ids = encoding['input_ids'].squeeze(0)  # Remove extra dimension
            attention_mask = encoding['attention_mask'].squeeze(0)

            # Pad labels to match the input sequence length
            labels = labels + [0] * (self.max_length - len(labels))  # Pad to max_length
            labels = torch.tensor(labels, dtype=torch.long)

            # Return a dictionary of tensors
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

    async def fine_tune_model(self):
        """
        Fine-tune the `Gepe55o/mountain-ner-bert-base` model on the `mountains-ner` dataset.
        """
        # Load the dataset and tokenizer
        dataset = await self.get_dataset()
        tokenizer = AutoTokenizer.from_pretrained('Gepe55o/mountain-ner-bert-base', ignore_mismatched_sizes=True)

        train_dataset = self.MountainNERDataset(dataset['train'], tokenizer)
        val_dataset = self.MountainNERDataset(dataset['test'], tokenizer)

        # Load pre-trained model
        model = AutoModelForTokenClassification.from_pretrained('Gepe55o/mountain-ner-bert-base', num_labels=5, ignore_mismatched_sizes=True)

        # Set up training arguments
        training_args: TrainingArguments = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=200,
            save_steps=500,
            gradient_accumulation_steps=2,
            fp16=torch.cuda.is_available(),
        )

        trainer: Trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained('./mountain-ner-finetuned')
        tokenizer.save_pretrained('./mountain-ner-finetuned')
        logger.info("Fine-tuned model saved at './mountain-ner-finetuned'")

    async def compute_similarity_and_rank(self):
        """
        Computes cosine similarity between mountain names based on BERT embeddings
        and ranks them by both frequency and semantic similarity.
        """
        # Load the fine-tuned model
        model = AutoModelForTokenClassification.from_pretrained('./mountain-ner-finetuned')
        tokenizer = AutoTokenizer.from_pretrained('./mountain-ner-finetuned')

        # Get dataset and prepare input sentences
        data_keys = await self.get_dataset()
        data_keys_split = [' '.join(item.split()) for item in data_keys['train']['sentence']]
        encodings = tokenizer(data_keys_split, padding=True, return_tensors='pt')

        logger.info(f'Encodings: {encodings.keys()}')

        with torch.no_grad():
            embeddings = model(**encodings)[0]

        # Get CLS embeddings and normalize them
        cls_embeddings = embeddings[:, 0, :]
        normalized_cls = F.normalize(cls_embeddings, p=2, dim=1)

        # Compute cosine similarity between all embeddings
        cls_dist = normalized_cls.matmul(normalized_cls.T)
        cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
        cls_dist = cls_dist.cpu().numpy()

        cls_similarity = cosine_similarity(cls_dist)

        # Rank the treatments based on cosine similarity
        treatment_freq = await self.treatment_frequency()
        treatments, frequencies = zip(*treatment_freq)

        top_n = 10
        mountains = treatments[:top_n]
        related_treatments = self.get_related_treatments(cls_similarity, mountains)

        # Save the results to a DataFrame
        data = {
            'Mountain (NER)': mountains,
            'Frequency': frequencies[:top_n],
        }

        result_df = pd.DataFrame(data)
        result_df.to_csv('MOUNTAINS_NER.csv', index=False)

        logger.info(f'Top mountains based on BERT similarity algorithm: {related_treatments}')
        self.visualize_top_treatments(mountains, frequencies[:top_n], related_treatments)

    def get_related_treatments(self, similarity_matrix, top_treatments):
        """
        Given the cosine similarity matrix and the list of top treatments,
        return the most related treatments.
        """
        num_treatments = len(top_treatments)
        related_treatments = {}
        for idx in range(num_treatments):
            similarity_scores = similarity_matrix[idx]
            related_indices = similarity_scores.argsort()[-4:-1][::-1]
            valid_related_indices = [i for i in related_indices if i < num_treatments]
            related_treatments[top_treatments[idx]] = [top_treatments[i] for i in valid_related_indices]
        return related_treatments

    def visualize_top_treatments(self, treatments, frequencies, related_treatments):
        """
        Visualizes the top treatments and their most related treatments.
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(treatments), y=list(frequencies), palette='Blues_d')
        plt.xticks(rotation=45)
        plt.title('Top 10 mountains')
        plt.xlabel('Mountains')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    async def treatment_frequency(self):
        """
        Computes and returns the frequency of treatment options discussed in the dataset.
        """
        data_keys = await self.get_dataset()
        data_keys_split = [item.split() for item in data_keys['train']['sentence']]
        key_phrases = [word for sublist in data_keys_split for word in sublist]
        frequency = Counter(key_phrases)
        sorted_frequency = frequency.most_common()
        return sorted_frequency

    async def running(self) -> None:
        """
        Runs the full workflow: fine-tune the model and compute similarity and rank.
        """
        await self.fine_tune_model()
        await self.compute_similarity_and_rank()
