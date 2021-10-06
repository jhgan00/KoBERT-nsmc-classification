import argparse
import random
import datetime

import torch.nn
import numpy as np
from kobert_transformers import get_tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from core.model import get_classifier
from core.trainer import Trainer
from core.data_loader import create_data_loader


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 데이터
    tokenizer = get_tokenizer()
    train_loader = create_data_loader(args.train, tokenizer, args.max_len, args.train_batch_size, shuffle=True)
    valid_loader = create_data_loader(args.valid, tokenizer, args.max_len, args.valid_batch_size)
    test_loader = create_data_loader(args.test, tokenizer, args.max_len, args.valid_batch_size)

    # 모델
    model = get_classifier(args.bert_model_name)
    model.to(args.device)
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

    # 모델 학습
    trainer = Trainer(model, train_loader, valid_loader, optimizer, scheduler, args)

    if args.mode == "train":
        print("=" * 10, "Configuration", "=" * 10)
        for arg in vars(args): print(f"{arg}={getattr(args, arg)}")
        print("=" * 34, end="\n\n")
        trainer.train()
    elif args.mode == "test":
        trainer.test(args.load_model, test_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
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
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)
