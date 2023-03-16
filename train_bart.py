from functools import partial

import torch
import evaluate as ev
from absl import app, flags
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import get_constant_schedule, get_constant_schedule_with_warmup
from tqdm.auto import tqdm

from data_preprocess import build_dataloader_fn


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.ckpt)
    model = BartForConditionalGeneration.from_pretrained(FLAGS.ckpt).to('cuda')
    optimizer = AdamW(model.parameters(), lr=FLAGS.lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, FLAGS.warmup)  # get_constant_schedule(optimizer)
    writer = SummaryWriter(FLAGS.save_dir)
    evaluate = partial(evaluate_fn, model=model, tokenizer=tokenizer, writer=writer)

    build_dataloader = build_dataloader_fn(model, tokenizer)
    train_dl = build_dataloader(data_split='train')
    eval_train_dl = build_dataloader(data_split='train', is_tgt_emo=lambda e: e == FLAGS.emo)
    eval_val_dl = build_dataloader(data_split='val', is_tgt_emo=lambda e: e == FLAGS.emo)

    model.train()
    optimizer.zero_grad()
    pbar = tqdm(total=FLAGS.n_steps)
    best_metric = float('-inf')

    for step, batch in enum_steps(train_dl, total_steps=FLAGS.n_steps*FLAGS.grad_acc):
        loss, *_ = model(**batch, return_dict=False)
        loss /= FLAGS.grad_acc
        loss.backward()

        if (step + 1) % FLAGS.grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            pbar.update(1)

        if (step + 1) % (FLAGS.eval_freq * FLAGS.grad_acc) == 0:
            opt_step = (step + 1) // FLAGS.grad_acc
            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], opt_step)
            model.eval()
            evaluate('train', eval_train_dl, opt_step)
            metric = evaluate('val', eval_val_dl, opt_step)
            if metric > best_metric:
                print(f'Saving best model with validation rougeL {metric:.3f}...')
                model.save_pretrained(writer.logdir, from_pt=True)
                best_metric = metric
            model.train()

    pbar.close()
    tokenizer.save_pretrained(writer.logdir, from_pt=True)


@torch.inference_mode()
def evaluate_fn(split, dataloader, step, model, tokenizer, writer):
    rouge = ev.load('rouge')
    losses = []

    for _, batch in tqdm(enum_steps(dataloader, total_steps=5), total=5):
        loss, *_ = model(**batch, return_dict=False)
        losses.append(loss)

        summary_ids = model.generate(batch['input_ids'], length_penalty=0.8, num_beams=8, max_length=128)
        preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        rouge.add_batch(predictions=preds, references=refs)

    metrics = rouge.compute()
    avg_loss = torch.mean(torch.stack(losses)).item()
    print(f'{split}: {avg_loss=:3f}, rougeL={metrics["rougeL"]:.3f}')
    
    writer.add_scalar(f'{split}/loss', avg_loss, step)
    for name, val in metrics.items():
        writer.add_scalar(f'{split}/metric/{name}', val, step)

    return metrics['rougeL']


def enum_steps(dataloader, total_steps):
    step = 0
    while True:
        for batch in dataloader:
            if step == total_steps:
                return
            batch = {k : v.to('cuda') for k, v in batch.items()}
            step += 1
            yield step, batch


if __name__ == '__main__':
    import os

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


    FLAGS = flags.FLAGS

    flags.DEFINE_string('ckpt', 'facebook/bart-base', 'Checkpoint name')
    flags.DEFINE_string('save_dir', None, 'Checkpoint name')

    flags.DEFINE_float('lr', 5e-5, 'Learning rate')
    flags.DEFINE_integer('warmup', None, 'Number of steps')

    flags.DEFINE_integer('batch_size', 16, 'Batch size')
    flags.DEFINE_integer('grad_acc', 1, 'Gradient accumulation steps')
    flags.DEFINE_integer('n_steps', None, 'Number of steps')
    flags.DEFINE_integer('eval_freq', None, 'Number of steps per evaluation')

    flags.DEFINE_string('emo', 'anger', 'Target emotion')

    flags.mark_flag_as_required('warmup')
    flags.mark_flag_as_required('n_steps')
    flags.mark_flag_as_required('eval_freq')
    flags.mark_flag_as_required('save_dir')

    app.run(main)
