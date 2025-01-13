import numpy

import pandas as pd
from collections import Counter

from datasets import load_dataset, DatasetDict

from loguru import logger

import torch
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer

from sklearn.metrics.pairwise import cosine_similarity

from src.core.consts import MOUNTAIN_MODEL, MOUNTAIN_DATASET


class MountainsNERSimple:

    async def data_keys(self) -> DatasetDict:
        """
        Fetches and returns the key phrases (treatment options) from preprocessed data.
        """
        dataset: DatasetDict = load_dataset(MOUNTAIN_DATASET)
        logger.info(dataset)
        return dataset

    async def mountains_frequency(self):
        """
        Computes and returns the frequency of treatment options discussed in the dataset.
        """
        data_keys: DatasetDict = await self.data_keys()
        data_keys_split: list[list[str]] = [item.split() for item in data_keys]
        key_phrases: list[str] = [word for sublist in data_keys_split for word in sublist]
        frequency = Counter(key_phrases)
        sorted_frequency = frequency.most_common()

        logger.info(f'Most common mountains: {sorted_frequency}')
        return sorted_frequency

    async def bert_similarity(self):
        """
        Calculates and returns the BERT embeddings for the treatment options.
        """
        # Load pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(MOUNTAIN_MODEL, ignore_mismatched_sizes=True)
        model = BertModel.from_pretrained(MOUNTAIN_MODEL, ignore_mismatched_sizes=True)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Fetch the data from the dataset
        dataset = await self.data_keys()

        # Extracting sentences or text from the dataset
        data_keys = [item['sentence'].split() for item in dataset['test']]  # or 'test', depending on the split needed

        # Check the first few entries to ensure it's in the correct format
        logger.info(f'First few entries: {data_keys[:5]}')

        # Now you can tokenize the data properly
        encodings = tokenizer(data_keys, padding=True, truncation=True, return_tensors='pt')

        logger.info(f'Encodings: {encodings.keys()}')
        logger.info([f'{tokens} ==> {tokenizer.convert_ids_to_tokens(tokens)}' for tokens in encodings['input_ids']])

        # Get the BERT embeddings
        with torch.no_grad():
            embeddings = model(**encodings)[0]

        logger.info(f'Embedding shape: {embeddings.shape}')
        return embeddings

    async def compute_similarity_and_rank(self):
        """
        Computes cosine similarity between treatment options based on BERT embeddings,
        and ranks them by both frequency and semantic similarity.
        """
        embeddings = await self.bert_similarity()

        cls_embeddings = embeddings[:, 0, :]
        normalized_cls = F.normalize(cls_embeddings, p=2, dim=1)

        cls_dist = normalized_cls.matmul(normalized_cls.T)
        cls_dist = cls_dist.new_ones(cls_dist.shape) - cls_dist
        cls_dist = cls_dist.cpu().numpy()

        cls_similarity = cosine_similarity(cls_dist)

        treatment_freq = await self.mountains_frequency()
        treatments, frequencies = zip(*treatment_freq)

        top_n = 10
        top_mountains = treatments[:top_n]

        # Create a DataFrame with mountains options, frequencies
        data = {
            'Mountains': top_mountains,
            'Frequency': frequencies[:top_n],
        }

        result_df = pd.DataFrame(data)
        # Save to CSV file
        result_df.to_csv('treatment_analysis.csv', index=False)
        logger.info(f'Top mountains based on BERT similarity: {result_df}')
