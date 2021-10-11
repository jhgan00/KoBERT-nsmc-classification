import argparse
import random
import datetime

import torch.nn
import numpy as np
import treform as ptm
from kobert_transformers import get_tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from core.model import get_classifier
from core.trainer import Trainer
from core.data_loader import create_data_loader
from core.predictor import Predictor


def main(args):

    print("=" * 10, "Configuration", "=" * 10)
    for arg in vars(args): print(f"{arg}={getattr(args, arg)}")
    print("=" * 34, end="\n\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    tokenizer = get_tokenizer()

    model = get_classifier(args.bert_model_name)
    if args.load_model is not None:
        state_dict = torch.load(args.load_model)
        model.load_state_dict(state_dict)
    model.to(args.device)

    if args.mode == "train":

        train_loader = create_data_loader(args.train, tokenizer, args.max_len, args.train_batch_size, shuffle=True)
        valid_loader = create_data_loader(args.valid, tokenizer, args.max_len, args.valid_batch_size)
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta_1, args.beta_2),
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=len(train_loader) * args.num_epochs
        )

        trainer = Trainer(model, train_loader, valid_loader, optimizer, scheduler, args)
        trainer.train()

    elif args.mode == "test":
        predictor = Predictor(model, tokenizer, max_len=args.max_len, device=args.device)
        test_loader = create_data_loader(args.test, tokenizer, args.max_len, args.valid_batch_size)
        predictor.test(test_loader)

    else:
        pipeline = None
        if args.preprocess == "lemma":
            pipeline = ptm.Pipeline(
                ptm.splitter.NLTK(),
                ptm.tokenizer.TwitterKorean(),
                ptm.lemmatizer.SejongPOSLemmatizer(),
                ptm.helper.SelectWordOnly(),
            )
        elif args.preprocess == "lemma_stopwords":
            pipeline = ptm.Pipeline(
                ptm.splitter.NLTK(),
                ptm.tokenizer.TwitterKorean(),
                ptm.lemmatizer.SejongPOSLemmatizer(),
                ptm.helper.SelectWordOnly(),
                ptm.helper.StopwordFilter(file="./stopwords/stopwordsKor.txt")
            )
        predictor = Predictor(model, tokenizer, pipeline=pipeline, max_len=args.max_len, device=args.device)
        predictor.interact()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test", "interactive"], default="train")
    parser.add_argument("--expr_name", type=str, default=str(datetime.datetime.now()))
    parser.add_argument("--train", type=str, default="./data/train.txt")
    parser.add_argument("--valid", type=str, default="./data/valid.txt")
    parser.add_argument("--test", type=str, default="./data/test.txt")
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--valid_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta_1", type=float, default=.9)
    parser.add_argument("--beta_2", type=float, default=.999)
    parser.add_argument("--weight_decay", type=float, default=0.5)
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--early_stopping_rounds", type=int, default=5)
    parser.add_argument("--bert_model_name", type=str, default="monologg/kobert")
    parser.add_argument("--load_model", type=str, required=False)
    parser.add_argument("--preprocess", type=str, choices=["no_preprocessing", "lemma", "lemma_stopwords"], default="no_preprocessing")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)
