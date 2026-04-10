import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch
import pandas as pd

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.

        Inputs:
            * data_folder (str): Path to folder containing data files
            * split (str): "train", "dev", or "test"
        '''
        # make sure split is valid
        if split not in ["train", "dev", "test"]:
            raise ValueError("split must be one of 'train', 'dev', or 'test'")
        self.data_folder = data_folder
        self.split = split
        # You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both the encoder and decoder output. 
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.decoder_start_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        """
        Tokenizes the data and populates self.examples with the tokenized examples. 
        Each example in self.examples should be a dictionary containing the following keys:
            * encoder_input: The tokenized input to be fed into the T5 encoder.
            * encoder_mask: Mask associated with padding tokens in the encoder input
            * decoder_input: Decoder input ids to be fed into T5 decoder (only for training and dev set)
            * decoder_target: The target tokens with which to train the decoder (only for training and dev set)
            * initial_decoder_input: The very first input token to be decoder (only to be used in evaluation)
        """
        self.examples = []
        # Class behavior should be different on the test set.
        # no decoder input/target for test set since we don't have access to target SQL queries
        if split == "test":
            # load_prompting_data() returns train_x, train_y, dev_x, dev_y, test_x
            _, _, _, _, test_x = load_prompting_data(data_folder)
            for x in test_x:
                # tokenizer returns a dict {input_ids: tensor, attention_mask: tensor, ...}
                # we just care about input_ids and attention_mask for the encoder
                # You want to provide the decoder some beginning of sentence token
                # Any extra-id on the T5Tokenizer should serve that purpose.
                encoder_tokens = tokenizer(x, return_tensors="pt", truncation=True)
                example = {
                    'encoder_input': encoder_tokens['input_ids'].squeeze(0),
                    'encoder_mask': encoder_tokens['attention_mask'].squeeze(0),
                    'initial_decoder_input': torch.tensor([self.decoder_start_id], dtype=torch.long),
                }
                self.examples.append(example)
        else:
            train_x, train_y, dev_x, dev_y, _ = load_prompting_data(data_folder)
            data_x = train_x if split == "train" else dev_x
            data_y = train_y if split == "train" else dev_y

            for x, y in zip(data_x, data_y):
                encoder_tokens = tokenizer(x, return_tensors="pt", truncation=True)
                target_ids = tokenizer(y, return_tensors="pt", truncation=True)['input_ids'].squeeze(0)
                initial_decoder_input = torch.tensor([self.decoder_start_id], dtype=torch.long)
                # use BOS + shifted target as decoder input
                decoder_input = torch.cat([initial_decoder_input, target_ids[:-1]], dim=0)

                # build example dict
                example = {
                    'encoder_input': encoder_tokens['input_ids'].squeeze(0),
                    'encoder_mask': encoder_tokens['attention_mask'].squeeze(0),
                    'decoder_input': decoder_input,
                    'decoder_target': target_ids,
                    'initial_decoder_input': initial_decoder_input,
                }
                # add to list of examples
                self.examples.append(example)
    def __len__(self):
        # number of examples in the dataset
        return len(self.examples)

    def __getitem__(self, idx):
        # returns the example at index idx in self.examples
        return self.examples[idx]

    def calc_dataset_statistics(self, stage="processed"):
        """
        Computes dataset stats before or after preprocessing.

        Inputs:
            * stage (str): "raw" or "processed"

        Returns:
            * pd.DataFrame with one row and columns:
              Number of examples, Mean sentence length, Mean SQL query length,
              Vocabulary size (natural language), Vocabulary size (SQL)
        """
        if stage not in ["raw", "processed"]:
            raise ValueError("stage must be one of 'raw' or 'processed'")

        if stage == "raw":
            return self._calc_raw_dataset_statistics()
        return self._calc_processed_dataset_statistics()

    def compare_dataset_statistics(self):
        """Return raw and processed statistics as two DataFrames."""
        raw_df = self._calc_raw_dataset_statistics()
        processed_df = self._calc_processed_dataset_statistics()
        return raw_df, processed_df

    def _calc_raw_dataset_statistics(self):
        train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(self.data_folder)
        if self.split == "train":
            data_x, data_y = train_x, train_y
        elif self.split == "dev":
            data_x, data_y = dev_x, dev_y
        else:
            data_x, data_y = test_x, None

        num_examples = len(data_x)
        nl_tokens = [nltk.word_tokenize(text) for text in data_x]
        mean_sentence_length = sum(len(tokens) for tokens in nl_tokens) / max(num_examples, 1)
        vocab_size_nl = len(set(token for tokens in nl_tokens for token in tokens))
        mean_sql_query_length = float('nan')
        vocab_size_sql = float('nan')
        if data_y is not None:
            sql_tokens = [nltk.word_tokenize(query) for query in data_y]
            mean_sql_query_length = sum(len(tokens) for tokens in sql_tokens) / max(num_examples, 1)
            vocab_size_sql = len(set(token for tokens in sql_tokens for token in tokens))

        return pd.DataFrame([
            {
                'Number of examples': num_examples,
                'Mean sentence length': mean_sentence_length,
                'Mean SQL query length': mean_sql_query_length,
                'Vocabulary size (natural language)': vocab_size_nl,
                'Vocabulary size (SQL)': vocab_size_sql,
            }
        ]).T

    def _calc_processed_dataset_statistics(self):
        num_examples = len(self.examples)
        mean_sentence_length = (
            sum(len(example['encoder_input']) for example in self.examples) / max(num_examples, 1)
        )
        vocab_size_nl = len(
            set(token_id.item() for example in self.examples for token_id in example['encoder_input'])
        )
        mean_sql_query_length = float('nan')
        vocab_size_sql = float('nan')
        if self.split != "test":
            mean_sql_query_length = (
                sum(len(example['decoder_target']) for example in self.examples) / max(num_examples, 1)
            )
            vocab_size_sql = len(
                set(token_id.item() for example in self.examples for token_id in example['decoder_target'])
            )

        return pd.DataFrame([
            {
                'Number of examples': num_examples,
                'Mean sentence length': mean_sentence_length,
                'Mean SQL query length': mean_sql_query_length,
                'Vocabulary size (natural language)': vocab_size_nl,
                'Vocabulary size (SQL)': vocab_size_sql,
            }
        ]).T


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''

    # get corresponding entry in dict for each example in batch
    encoder_ids = [example['encoder_input'] for example in batch]
    encoder_mask = [example['encoder_mask'] for example in batch]
    decoder_inputs = [example['decoder_input'] for example in batch]
    decoder_targets = [example['decoder_target'] for example in batch]
    initial_decoder_inputs = [example['initial_decoder_input'] for example in batch]

    # pad to the max length in the batch and stack into tensors of shape BxT
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''

    # same as above but without decoder inputs/targets
    encoder_ids = [example['encoder_input'] for example in batch]
    encoder_mask = [example['encoder_mask'] for example in batch]
    initial_decoder_inputs = [example['initial_decoder_input'] for example in batch]

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    initial_decoder_inputs = pad_sequence(initial_decoder_inputs, batch_first=True, padding_value=PAD_IDX)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    """Build dataloader for split"""
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    """Load data for each split"""
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    """Load lines from a text file and strip whitespace"""
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    """Load text-to-SQL data from canonical .nl/.sql files.

    Returns:
        train_x, train_y, dev_x, dev_y, test_x
    where x is natural language and y is SQL.
    """
    # follow structure in data folder
    train_nl_path = os.path.join(data_folder, 'train.nl')
    train_sql_path = os.path.join(data_folder, 'train.sql')
    dev_nl_path = os.path.join(data_folder, 'dev.nl')
    dev_sql_path = os.path.join(data_folder, 'dev.sql')
    test_nl_path = os.path.join(data_folder, 'test.nl')

    # if they all exist, load and return
    if all(os.path.exists(p) for p in [train_nl_path, train_sql_path, dev_nl_path, dev_sql_path, test_nl_path]):
        train_x = load_lines(train_nl_path)
        train_y = load_lines(train_sql_path)
        dev_x = load_lines(dev_nl_path)
        dev_y = load_lines(dev_sql_path)
        test_x = load_lines(test_nl_path)
        return train_x, train_y, dev_x, dev_y, test_x
    
    return train_x, train_y, dev_x, dev_y, test_x