from absl import app, flags
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import AutoTokenizer, BartForConditionalGeneration
from tqdm.auto import tqdm

from data_preprocess import EMO_LIST, build_dataloader_fn
from eval_bart import evaluate


def main(argv):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.ckpt)
    model = BartForConditionalGeneration.from_pretrained(FLAGS.ckpt).to('cuda')
    optimizer = AdamW(model.parameters(), lr=FLAGS.lr)

    build_dataloader = build_dataloader_fn(model, tokenizer)
    train_dl = build_dataloader(data_split='train')
    train_emo_dl = build_dataloader(data_split='train', is_tgt_emo=lambda e: e == FLAGS.emo)
    val_emo_dl = build_dataloader(data_split='val', is_tgt_emo=lambda e: e == FLAGS.emo)

    writer = SummaryWriter(FLAGS.save_dir)
    model.train()

    def log_metrics(split, metrics, step):
        for name, val in metrics.items():
            writer.add_scalar(f'{split}/metric/{name}', val, step)

    for step, batch in enum_steps(train_dl):
        loss, *_ = model(**batch, return_dict=False)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % FLAGS.eval_freq == 0:
            train_loss, train_metrics = evaluate(model, tokenizer, train_emo_dl)
            writer.add_scalar('train/loss', train_loss, step)
            log_metrics('train', train_metrics, step)

            val_loss, val_metrics = evaluate(model, tokenizer, val_emo_dl)
            writer.add_scalar('val/loss', val_loss, step)
            log_metrics('val', val_metrics, step)

            model.train()

    tokenizer.save_pretrained(writer.logdir, from_pt=True)
    model.save_pretrained(writer.logdir, from_pt=True)


def enum_steps(dl):
    def inf_loop():
        while True:
            for batch in dl:
                batch = {k : v.to('cuda') for k, v in batch.items()}
                yield batch

    step_iter = zip(range(FLAGS.n_steps), inf_loop())

    return tqdm(step_iter, total=FLAGS.n_steps)


if __name__ == '__main__':
    import os

    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    FLAGS = flags.FLAGS

    flags.DEFINE_string('ckpt', 'facebook/bart-base', 'Checkpoint name')
    flags.DEFINE_string('save_dir', None, 'Checkpoint name')
    flags.DEFINE_float('lr', 5e-5, 'Learning rate')
    flags.DEFINE_integer('batch_size', 16, 'Batch size')
    flags.DEFINE_integer('n_steps', None, 'Number of steps')
    flags.DEFINE_integer('eval_freq', None, 'Number of steps per evaluation')
    flags.DEFINE_string('emo', 'anger', 'Target emotion')

    flags.mark_flag_as_required('n_steps')
    flags.mark_flag_as_required('eval_freq')
    flags.mark_flag_as_required('save_dir')

    app.run(main)
