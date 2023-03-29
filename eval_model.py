import torch
import evaluate as ev
import numpy as np
from absl import flags
from tqdm.auto import tqdm

from preprocess_data import EMO_LIST

FLAGS = flags.FLAGS


@torch.inference_mode()
def evaluate(model, tokenizer, eval_dls, compute_loss=False):
    rouge_l = dict()
    accum_loss = 0

    for emo in tqdm(EMO_LIST):
        rouge = ev.load('rouge')

        for batch in eval_dls[emo]:
            batch = {k : v.to('cuda') for k, v in batch.items()}

            summary_ids = model.generate(batch['input_ids'], length_penalty=0.8, num_beams=8, max_length=128)
            preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            rouge.add_batch(predictions=preds, references=refs)

            if compute_loss:
                loss, *_ = model(**batch, return_dict=False)
                accum_loss += loss.detach().item()

        rouge_l[emo] = np.around(rouge.compute(rouge_types=['rougeL'])['rougeL'], 4)

    avg_rouge = np.mean([*rouge_l.values()])

    if compute_loss:
        n_steps = len(eval_dls[EMO_LIST[0]]) * len(EMO_LIST)
        avg_loss = accum_loss / n_steps
        return rouge_l, avg_rouge, avg_loss

    return rouge_l, avg_rouge


def make_eval_dataloaders(data_dict, dd2dl):
    n_samples = len(data_dict['emo'])
    eval_dls = dict()

    for emo in EMO_LIST:
        emo_dd = {'post': [], 'emo': [], 'summ': []}
        for i in range(n_samples):
            if data_dict['emo'][i] == emo:
                emo_dd['post'].append(data_dict['post'][i])
                emo_dd['emo'].append(emo)
                emo_dd['summ'].append(data_dict['summ'][i])
        eval_dls[emo] = dd2dl(emo_dd)

    return eval_dls


def main(argv):
    rng = set_randomness(FLAGS.seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAGS.ckpt).to('cuda')
    if FLAGS.ckpt.startswith('t5'):
        tokenizer = AutoTokenizer.from_pretrained(FLAGS.ckpt, model_max_length=512)
    else:
        tokenizer = AutoTokenizer.from_pretrained(FLAGS.ckpt)

    make_dataset = config_dataset(tokenizer)
    make_dataloader = config_dataloader(model, tokenizer, rng)
    dd2dl = lambda dd: make_dataloader(make_dataset(dd))
    data_dict = data_dict_allsumm(FLAGS.split, concat_same_emo=True)
    eval_dls = make_eval_dataloaders(data_dict, dd2dl)

    log_print = config_log_print(f'{FLAGS.ckpt}/{FLAGS.split}.log')
    rouge, avg_rouge = evaluate(model, tokenizer, eval_dls)
    log_print(f'ROUGE-L={rouge}, {avg_rouge=:.4f}')


if __name__ == '__main__':
    import os
    from absl import app
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    from preprocess_data import (
        data_dict_allsumm, config_dataset, config_dataloader
    )
    from utils import set_randomness, config_log_print

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    FLAGS = flags.FLAGS
    flags.DEFINE_integer('seed', None, 'Random seed', required=True)
    flags.DEFINE_integer('batch_size', None, 'Batch size', required=True)
    flags.DEFINE_string('ckpt', None, 'Checkpoint name', required=True)
    flags.DEFINE_string('split', None, 'Data split', required=True)

    app.run(main)
