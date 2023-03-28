from pathlib import Path

from absl import flags
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import get_constant_schedule, get_constant_schedule_with_warmup
from tqdm.auto import tqdm

from eval_model import evaluate, make_eval_dataloaders
from preprocess_data import (
    EMO_LIST, set_randomness,
    data_dict_allsumm, data_dict_balanced,
    config_dataset, config_dataloader
)

FLAGS = flags.FLAGS


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
    train_dl = dd2dl(select_train_data_dict(FLAGS.dd))
    eval_train_dls = make_eval_dataloaders('train', dd2dl)
    eval_val_dls = make_eval_dataloaders('val', dd2dl)

    optimizer = AdamW(model.parameters(), lr=FLAGS.lr)
    if FLAGS.warmup is not None:
        scheduler = get_constant_schedule_with_warmup(optimizer, FLAGS.warmup)
    else:
        scheduler = get_constant_schedule(optimizer)

    save_dir = Path('new_runs') / FLAGS.exp_name
    assert not save_dir.exists(), f'{save_dir} already exist.'
    writer = SummaryWriter(save_dir)

    model.train()
    optimizer.zero_grad()
    pbar = tqdm(total=FLAGS.train_steps)
    forward_steps = FLAGS.train_steps * FLAGS.grad_acc
    best_rouge = float('-inf')
    accum_loss = 0

    for step, batch in zip(range(forward_steps), load_batch(train_dl)):
        loss, *_ = model(**batch, return_dict=False)
        loss /= FLAGS.grad_acc
        loss.backward()
        accum_loss += loss.detach().item()

        if (step + 1) % FLAGS.grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            pbar.update(1)

        if (step + 1) % (FLAGS.eval_freq * FLAGS.grad_acc) == 0:
            opt_step = (step + 1) // FLAGS.grad_acc

            writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], opt_step)
            writer.add_scalar('train/loss', accum_loss/FLAGS.eval_freq, opt_step)
            accum_loss = 0

            rouge_train, rouge_train_avg = evaluate(model, tokenizer, eval_train_dls)
            print(f'Step {opt_step}: {rouge_train=}, {rouge_train_avg=:4f}')
            writer.add_scalar('train/ROUGE-L/avg', rouge_train_avg, opt_step)

            rouge_val, rouge_val_avg, val_loss = evaluate(model, tokenizer, eval_val_dls, compute_loss=True)
            print(f'Step {opt_step}: {rouge_val=}, {rouge_val_avg=:4f}')
            writer.add_scalar('val/ROUGE-L/avg', rouge_val_avg, opt_step)
            writer.add_scalar('val/loss', val_loss, opt_step)

            for emo in EMO_LIST:
                writer.add_scalar(f'train/ROUGE-L/{emo}', rouge_train[emo], opt_step)
                writer.add_scalar(f'val/ROUGE-L/{emo}', rouge_val[emo], opt_step)

            if rouge_val_avg > best_rouge:
                best_rouge = rouge_val_avg
                print(f'Saving best model with validation ROUGE-L {best_rouge:.4f}...')
                model.save_pretrained(save_dir, from_pt=True)

    pbar.close()
    writer.close()
    tokenizer.save_pretrained(save_dir, from_pt=True)


def select_train_data_dict(name):
    if name == 'balanced':
        return data_dict_balanced('train', FLAGS.seed)
    elif name == 'allsumm':
        return data_dict_allsumm('train', FLAGS.seed, concat_same_emo=False)
    elif name == 'allsumm_concat':
        return data_dict_allsumm('train', FLAGS.seed, concat_same_emo=True)
    else:
        raise ValueError(f'Invalid data dict enum {name}.')


def load_batch(dataloader):
    while True:
        for batch in dataloader:
            batch = {k : v.to('cuda') for k, v in batch.items()}
            yield batch


if __name__ == '__main__':
    import os
    from absl import app

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    flags.DEFINE_integer('seed', None, 'Random seed', required=True)
    flags.DEFINE_string('ckpt', None, 'Checkpoint name', required=True)
    flags.DEFINE_string('exp_name', None, 'Experiment name', required=True)

    flags.DEFINE_enum(
        'dd', None, ['balanced', 'allsumm', 'allsumm_concat'],
        'Data dictionary configuration', required=True
    )
    flags.DEFINE_integer('eval_freq', None, 'Number of steps between each evaluation', required=True)

    flags.DEFINE_integer('train_steps', None, 'Number of training steps', required=True)
    flags.DEFINE_float('lr', None, 'Learning rate', required=True)
    flags.DEFINE_integer('warmup', None, 'Number of warm-up steps')

    flags.DEFINE_integer('batch_size', None, 'Batch size', required=True)
    flags.DEFINE_integer('grad_acc', 1, 'Number of gradient accumulation steps')

    app.run(main)
