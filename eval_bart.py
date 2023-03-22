import torch
import evaluate as ev
from absl import app, flags
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

from data_preprocess import build_dataset, build_dataloader_fn, EMO_LIST


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(FLAGS.ckpt).to('cuda')

    def build_emo_dl_fn():
        build_dataloader = build_dataloader_fn(model, tokenizer)
        def build_emo_dl(emo):
            filter_fn = lambda e: e == emo
            dl = build_dataloader(data_split=FLAGS.split, is_tgt_emo=filter_fn)
            return dl
        return build_emo_dl

    @torch.inference_mode()
    def evaluate(dataloader):
        rouge = ev.load('rouge')
        model.eval()
        for batch in tqdm(dataloader):
            batch = {k : v.to('cuda') for k, v in batch.items()}
            summary_ids = model.generate(batch['input_ids'], length_penalty=0.8, num_beams=8, max_length=128)
            preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            rouge.add_batch(predictions=preds, references=refs)
        metrics = rouge.compute()
        print(f'{metrics=}')

    build_emo_dl = build_emo_dl_fn()

    for emo in EMO_LIST:
        print(f'Evaluating on emotion {emo}')
        emo_dl = build_emo_dl(emo)
        evaluate(emo_dl)


if __name__ == '__main__':
    import os

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 32, 'Batch size')
    flags.DEFINE_string('ckpt', 'facebook/bart-base', 'Checkpoint name')
    flags.DEFINE_string('split', 'val', 'Data split')

    app.run(main)
