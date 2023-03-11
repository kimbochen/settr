import os
import json 
from itertools import chain
from pathlib import Path

from absl import app, flags
from torch.optim import AdamW
from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm.auto import tqdm

from data_preprocess import EMO_LIST, build_dataloader_fn


os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', 'facebook/bart-base', 'Checkpoint name')
flags.DEFINE_float('lr', 5e-5, 'Learning rate')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('n_steps', None, 'Number of steps')
flags.DEFINE_integer('grad_acc', 1, 'Number of gradient accumulation steps')
flags.DEFINE_integer('n_epochs', None, 'Number of epochs')


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.ckpt)
    model = BartForConditionalGeneration.from_pretrained(FLAGS.ckpt).to('cuda')
    build_dataloader = build_dataloader_fn(model, tokenizer)
    train_dl = build_dataloader(data_split='train')

    optimizer = AdamW(model.parameters(), lr=FLAGS.lr)
    model.train()

    with tqdm(total=FLAGS.n_steps) as pbar:
        for step, batch in enum_steps(train_dl, FLAGS.n_steps*FLAGS.grad_acc):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            if (step + 1) % FLAGS.grad_acc == 0:
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_description(f'Loss: {loss:.4f}')
                pbar.update(1)

    save_ckpt = f'runs/bart-large-cnn-lr{FLAGS.lr:.2e}-step{FLAGS.n_steps}'
    tokenizer.save_pretrained(save_ckpt, from_pt=True)
    model.save_pretrained(save_ckpt, from_pt=True)


def enum_steps(dl, n_steps):
    def inf_loop():
        while True:
            for batch in dl:
                batch = {k : v.to('cuda') for k, v in batch.items()}
                yield batch
    return zip(range(n_steps), inf_loop())


if __name__ == '__main__':
    app.run(main)
