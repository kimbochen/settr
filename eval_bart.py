import torch
import evaluate as ev
from absl import app, flags
from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm.auto import tqdm

from data_preprocess import build_dataset, build_dataloader_fn, EMO_LIST


@torch.inference_mode()
def evaluate(model, tokenizer, dataloader, n_steps=5):
    rouge = ev.load('rouge')
    losses = []
    model.eval()

    for _, batch in tqdm(zip(range(n_steps), dataloader), total=n_steps):
        batch = {k : v.to('cuda') for k, v in batch.items()}

        loss, *_ = model(**batch, return_dict=False)
        losses.append(loss.detach().to('cpu'))

        summary_ids = model.generate(batch['input_ids'], length_penalty=0.8, num_beams=8, max_length=128)
        preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        rouge.add_batch(predictions=preds, references=refs)

    metrics = rouge.compute()
    avg_loss = torch.mean(torch.stack(losses)).item()
    print(f'{avg_loss=:3f}, {metrics=}')

    return avg_loss, metrics


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.ckpt)
    model = BartForConditionalGeneration.from_pretrained(FLAGS.ckpt).to('cuda')
    build_dataloader = build_dataloader_fn(model, tokenizer)

    for emo in EMO_LIST:
        val_dl = build_dataloader(data_split=FLAGS.split, is_tgt_emo=lambda e: e == emo)
        print(f'Evaluating on emotion {emo}')
        evaluate(model, tokenizer, val_dl)


if __name__ == '__main__':
    import os

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 32, 'Batch size')
    flags.DEFINE_string('ckpt', 'facebook/bart-base', 'Checkpoint name')
    flags.DEFINE_string('split', 'val', 'Data split')

    app.run(main)
