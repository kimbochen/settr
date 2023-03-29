# get_raw_dataset
import json
import random
from collections import defaultdict
from itertools import chain
from pathlib import Path

# data_dict_balanced
from collections import Counter, defaultdict

# config_dataset
from datasets import Dataset

# config_dataloader
import random
from os import sched_getaffinity
import torch
import numpy as np
from absl import flags
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

# data_dict_balanced
EMO_LIST = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'trust', 'anticipation']
FLAGS = flags.FLAGS  # config_dataloader


def get_raw_dataset(split, seed, concat_same_emo=True):
    '''
    Raw dataset format:
    raw_dataset: [sample]
    sample     : { 'post': str, 'annos': [anno] }
    anno       : { 'emo': str, 'summ': str }
    '''
    data_dir = Path('data/train_val_test-WITH_POSTS')
    assert data_dir.exists(), 'Data not in the correct file path.'

    data_path = data_dir / f'{split}_anonymized-WITH_POSTS.json'
    assert data_path.exists(), f'Cannot find {split} data split at {data_path}.'

    with data_path.open() as f:
        json_data = json.load(f)

    raw_dataset = []

    if concat_same_emo:
        for raw_sample in json_data.values():
            emo2summ = defaultdict(str)
            for anno in chain(*raw_sample['Annotations'].values()):
                if anno['Emotion'] != 'NA':
                    emo2summ[anno['Emotion']] += ' ' + anno['Abstractive']

            sample = {'post': raw_sample['Reddit Post'], 'annos': []}
            for emo, summ in emo2summ.items():
                anno = {'emo': emo, 'summ': summ}
                sample['annos'].append(anno)
            raw_dataset.append(sample)
    else:
        for raw_sample in json_data.values():
            sample = {'post': raw_sample['Reddit Post'], 'annos': []}
            for anno in chain(*raw_sample['Annotations'].values()):
                if anno['Emotion'] != 'NA':
                    emo_summ = {'emo': anno['Emotion'], 'summ': anno['Abstractive']}
                    sample['annos'].append(emo_summ)
            raw_dataset.append(sample)

    random.seed(seed)
    random.shuffle(raw_dataset)

    return raw_dataset


def data_dict_allsumm(split, **kwargs):
    raw_dataset = get_raw_dataset(split, FLAGS.seed, **kwargs)
    data_dict = {'post': [], 'emo': [], 'summ': []}

    for sample in raw_dataset:
        for anno in sample['annos']:
            data_dict['post'].append(sample['post'])
            data_dict['emo'].append(anno['emo'])
            data_dict['summ'].append(anno['summ'])

    return data_dict


def data_dict_balanced(split, sample_size=None):
    raw_dataset = get_raw_dataset(split, FLAGS.seed, concat_same_emo=True)
    data_dict = {'post': [], 'emo': [], 'summ': []}

    n_samples = dict.fromkeys(EMO_LIST, 0)
    sampling_emos = set(EMO_LIST)
    emo_freq = Counter(data_dict_allsumm(split, concat_same_emo=True)['emo'])
    if sample_size is None:
        sample_size = min(emo_freq.values())

    for sample in raw_dataset:
        annos = list(filter(lambda es: es['emo'] in sampling_emos, sample['annos']))
        if annos:
            anno = min(annos, key=lambda anno: emo_freq[anno['emo']])
            data_dict['post'].append(sample['post'])
            data_dict['emo'].append(anno['emo'])
            data_dict['summ'].append(anno['summ'])

            emo = anno['emo']
            n_samples[emo] += 1
            if n_samples[emo] == sample_size:
                sampling_emos.remove(emo)

    return data_dict


def config_dataset(tokenizer):
    instr = 'Generate a summary of what triggered {} in this post: {}'

    def verify_data_dict(dd):
        key_set = ['post', 'emo', 'summ']
        assert list(dd.keys()) == key_set, f'Invalid key set: {dd.keys()}'
        len_dict = {k: len(dd[k]) for k in key_set}
        assert len_dict['post'] == len_dict['emo'] == len_dict['summ'], f'{len_dict=}'

    def make_prompt(sample):
        return {'prompt': instr.format(sample['emo'], sample['post'])}

    def tokenize(sample):
        inputs = tokenizer(
            sample['prompt'],
            max_length=512, truncation=True, padding='max_length'
        )
        labels = tokenizer(
            sample['summ'], return_attention_mask=False,
            max_length=128, truncation=True, padding='max_length'
        )
        return {**inputs, 'labels': labels['input_ids']}

    def make_dataset(data_dict):
        verify_data_dict(data_dict)
        dataset = Dataset.from_dict(data_dict)
        dataset = dataset.map(make_prompt)
        dataset = dataset.remove_columns(['post', 'emo'])
        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.remove_columns(['prompt', 'summ'])
        dataset.set_format('torch')
        return dataset

    return make_dataset


def config_dataloader(model, tokenizer, rng, **kwargs):
    collator = DataCollatorForSeq2Seq(tokenizer, model, padding='longest')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    dl_kwargs = dict(
        collate_fn=collator, batch_size=FLAGS.batch_size,
        num_workers=len(sched_getaffinity(0)),
        worker_init_fn=seed_worker, generator=rng
    )
    dl_kwargs.update(kwargs)

    make_dataloader = lambda dataset: DataLoader(dataset, **dl_kwargs)

    return make_dataloader


def unit_test(argv):
    import os
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from utils import set_randomness

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    rng = set_randomness(FLAGS.seed)

    ckpt = 'facebook/bart-base'
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    make_dataset = config_dataset(tokenizer)
    make_dataloader = config_dataloader(model, tokenizer, rng, batch_size=32)

    data_dict = data_dict_balanced('train')
    dataset = make_dataset(data_dict)
    dataloader = make_dataloader(dataset)
    sb = next(iter(dataloader))
    print({k : v.shape for k, v in sb.items()})


if __name__ == '__main__':
    from absl import app
    flags.DEFINE_integer('seed', 3985, 'Batch size')
    flags.DEFINE_integer('batch_size', 16, 'Batch size')
    app.run(unit_test)
