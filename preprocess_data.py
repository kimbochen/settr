import json 
from collections import defaultdict
from itertools import chain
from pathlib import Path

from absl import flags
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


EMO_LIST = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'trust', 'anticipation']
DATA_DIR = Path('data/train_val_test-WITH_POSTS')
FLAGS = flags.FLAGS


def set_deterministic(seed=3985):
    import torch
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(seed)

    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return g, seed_worker


def build_data_dict(data_split, is_tgt_emo=lambda _: True):
    data_path = DATA_DIR / f'{data_split}_anonymized-WITH_POSTS.json'
    assert data_path.exists()

    with open(data_path) as f:
        data = json.load(f)

    data_dict = {'input': [], 'summary': []}
    input_str = 'Generate a summary of what triggered {} in this post: {}'

    for sample in data.values():
        emo_summ = defaultdict(str)

        for anno in chain(*sample['Annotations'].values()):
            if is_tgt_emo(anno['Emotion']) and anno['Emotion'] != 'NA':
                emo_summ[anno['Emotion']] += anno['Abstractive']

        for emo, summ in emo_summ.items():
            data_dict['input'].append(input_str.format(emo, sample['Reddit Post']))
            data_dict['summary'].append(summ)

    return data_dict


def build_balanced_train_data(**kwargs):
    data_path = DATA_DIR / f'train_anonymized-WITH_POSTS.json'
    assert data_path.exists()

    with open(data_path) as f:
        data = json.load(f)

    data_dict = {'input': [], 'summary': []}
    instr = 'Generate a summary of what triggered {} in this post: {}'
    emo_cnt = dict.fromkeys(EMO_LIST, 0)
    emo_list = set(EMO_LIST)

    emo_freq = {
        'anger': 470, 'disgust': 192, 'fear': 765, 'joy': 134,
        'sadness': 360, 'trust': 99, 'anticipation': 873
    }
    max_size = min(emo_freq.values())

    for sample in data.values():
        annos = chain(*sample['Annotations'].values())
        annos = list(filter(lambda a: a['Emotion'] in emo_list, annos))
        anno = min(annos, key=lambda a: emo_freq[a['Emotion']], default=None)
        if anno is not None:
            emo = anno['Emotion']
            data_dict['input'].append(instr.format(emo, sample['Reddit Post']))
            data_dict['summary'].append(anno['Abstractive'])
            emo_cnt[emo] += 1
            if emo_cnt[emo] == max_size:
                emo_list.remove(emo)

    return data_dict


def build_dataset(tokenizer, **dd_kwargs):
    if dd_kwargs['data_split'] == 'train':
        data_dict = build_balanced_train_data(**dd_kwargs)
    else:
        data_dict = build_data_dict(**dd_kwargs)

    raw_dataset = Dataset.from_dict(data_dict)

    def tokenize(sample):
        inputs = tokenizer(
            sample['input'],
            max_length=512, truncation=True, padding='max_length'
        )
        labels = tokenizer(
            sample['summary'], return_attention_mask=False,
            max_length=128, truncation=True, padding='max_length'
        )
        return {**inputs, 'labels': labels['input_ids']}

    dataset = raw_dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(['input', 'summary'])
    dataset.set_format('torch')

    return dataset


def build_dataloader_fn(model, tokenizer):
    collator = DataCollatorForSeq2Seq(tokenizer, model, padding='longest')
    dl_kwargs = dict(collate_fn=collator, batch_size=FLAGS.batch_size, num_workers=4)

    def build_dataloader(**dd_kwargs):
        dataset = build_dataset(tokenizer, **dd_kwargs)
        dl = DataLoader(dataset, **dl_kwargs)
        return dl

    return build_dataloader


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
