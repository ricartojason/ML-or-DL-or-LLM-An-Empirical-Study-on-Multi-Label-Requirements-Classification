#encoding:utf-8
import torch
import numpy as np
from pathlib import Path
from ..common.tools import logger, load_pickle
from ..callback.progressbar import ProgressBar
from torch.utils.data import TensorDataset
from ..io.bert_processor import InputExample, InputFeature

class GloveProcessor(object):
    """Base class for data converters for sequence classification data sets using GloVe embeddings."""

    def __init__(self, glove_path):
        """
        Initialize the GloveProcessor.
        
        Args:
            glove_path: Path to the GloVe embeddings file
        """
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.vocab = self._build_vocab(glove_path)
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)

    def _build_vocab(self, glove_path):
        """Build vocabulary from GloVe file."""
        vocab = {self.pad_token: 0, self.unk_token: 1}  # 初始化特殊token
        
        # 从GloVe文件中提取词表
        with open(glove_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                word = line.split()[0]
                vocab[word] = len(vocab)  # 为每个词分配一个唯一的索引
        
        return vocab

    def get_train(self, data_file):
        """Gets a collection of `InputExample`s for the train set."""
        return self.read_data(data_file)

    def get_dev(self, data_file):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.read_data(data_file)

    def get_test(self, lines):
        return lines

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ['Usa', 'Sup', 'Dep', 'Per']

    @classmethod
    def read_data(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        if 'pkl' in str(input_file):
            lines = load_pickle(input_file)
        else:
            lines = input_file
        return lines

    def truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncate a sequence pair to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def create_examples(self, lines, example_type, cached_examples_file):
        """Creates examples for data."""
        pbar = ProgressBar(n_total=len(lines), desc='create examples')
        if cached_examples_file.exists():
            logger.info("Loading examples from cached file %s", cached_examples_file)
            examples = torch.load(cached_examples_file)
        else:
            examples = []
            for i, line in enumerate(lines):
                guid = '%s-%d' % (example_type, i)
                text_a = line[0]
                label = line[1]
                if isinstance(label, str):
                    label = [np.float(x) for x in label.split(",")]
                else:
                    label = [np.float(x) for x in list(label)]
                text_b = None
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
                pbar(step=i)
            logger.info("Saving examples into cached file %s", cached_examples_file)
            torch.save(examples, cached_examples_file)
        return examples

    def tokenize(self, text):
        """Tokenize text into words."""
        return text.lower().split()

    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to their corresponding ids."""
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def create_features(self, examples, max_seq_len, cached_features_file):
        """Create features from examples."""
        pbar = ProgressBar(n_total=len(examples), desc='create features')
        if cached_features_file.exists():
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            features = []
            for ex_id, example in enumerate(examples):
                # Tokenize text
                tokens_a = self.tokenize(example.text_a)
                tokens_b = None
                label_id = example.label

                if example.text_b:
                    tokens_b = self.tokenize(example.text_b)
                    self.truncate_seq_pair(tokens_a, tokens_b, max_length=max_seq_len - 2)
                else:
                    if len(tokens_a) > max_seq_len:
                        tokens_a = tokens_a[:max_seq_len]

                # Convert tokens to ids
                input_ids = self.convert_tokens_to_ids(tokens_a)
                input_mask = [1] * len(input_ids)
                input_len = len(input_ids)

                # Add padding
                padding = [self.pad_token_id] * (max_seq_len - len(input_ids))
                input_ids += padding
                input_mask += [0] * (max_seq_len - len(input_mask))

                assert len(input_ids) == max_seq_len
                assert len(input_mask) == max_seq_len

                if ex_id < 2:
                    logger.info("*** Example ***")
                    logger.info(f"guid: {example.guid}")
                    logger.info(f"tokens: {' '.join(tokens_a)}")
                    logger.info(f"input_ids: {' '.join([str(x) for x in input_ids])}")
                    logger.info(f"input_mask: {' '.join([str(x) for x in input_mask])}")

                feature = InputFeature(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=[0] * max_seq_len,  # GloVe doesn't use segment_ids
                    label_id=label_id,
                    input_len=input_len
                )
                features.append(feature)
                pbar(step=ex_id)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
        return features

    def create_dataset(self, features, is_sorted=False):
        """Convert features to TensorDataset."""
        if is_sorted:
            logger.info("sorted data by the length of input")
            features = sorted(features, key=lambda x: x.input_len, reverse=True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lens)
        return dataset 