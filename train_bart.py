import os
import json 
from itertools import chain
from pathlib import Path

from absl import app, flags
from torch.optim import AdamW
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
from transformers import get_constant_schedule
from tqdm.auto import tqdm

from data_preprocess import build_dataset


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
    optim = AdamW(model.parameters(), lr=FLAGS.lr)
    sched = get_constant_schedule(optim)

    def compute_metrics(eval_preds):
        summ_ids, label_ids = eval_preds
        summ = tokenizer.batch_decode(summ_ids, skip_special_tokens=True)
        labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return metric.compute(predictions=summs, references=labels)

    train_args = TrainingArguments(
        f'runs2/bart-base-lr{FLAGS.lr:.2e}-{FLAGS.n_epochs}epochs',
        evaluation_strategy='epoch',
        per_device_train_batch_size=FLAGS.batch_size,
        per_device_eval_batch_size=FLAGS.batch_size,
        gradient_accumulation_steps=FLAGS.grad_acc,
        learning_rate=FLAGS.lr,
        num_train_epochs=FLAGS.n_epochs,
        logging_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding='longest'),
        train_dataset=build_dataset(tokenizer, data_split='train'),
        eval_dataset=build_dataset(tokenizer, data_split='val'),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=[optim, sched]
    )

    trainer.train()


if __name__ == '__main__':
    app.run(main)
