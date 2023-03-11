import evaluate as ev
from absl import app, flags
from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm.auto import tqdm

from data_preprocess import build_dataset, build_dataloader_fn, EMO_LIST


def evaluate(model, tokenizer, dataloader):
    metric = ev.load('rouge')
    model.eval()

    for batch in tqdm(dataloader):
        summary_ids = model.generate(
            batch['input_ids'].to('cuda'),
            length_penalty=0.8, num_beams=8, max_length=128
        )
        preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        metric.add_batch(predictions=preds, references=refs)

    print(metric.compute())


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.ckpt)
    model = BartForConditionalGeneration.from_pretrained(FLAGS.ckpt).to('cuda')
    build_dataloader = build_dataloader_fn(model, tokenizer)

    for emo in EMO_LIST:
        val_dl = build_dataloader(data_split='val', is_tgt_emo=lambda e: e == emo)
        print(f'Evaluating on emotion {emo}')
        evaluate(model, tokenizer, val_dl)


if __name__ == '__main__':
    import os

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 32, 'Batch size')
    flags.DEFINE_string('ckpt', 'facebook/bart-base', 'Checkpoint name')

    app.run(main)
