import os
import copy
import json
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, model, train_loader, valid_loader, optimizer, scheduler, args):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.global_step = 0
        self.eval_accuracy = 0
        self.best_model = None
        self.early_stopping_counter = 0

        self.args = args
        self.expr_name = args.expr_name
        self.num_epochs = args.num_epochs
        self.eval_every = args.eval_every
        self.early_stopping_rounds = args.early_stopping_rounds
        self.device = args.device

        self.model_dir = os.path.join("./models", self.expr_name)
        self.log_dir = os.path.join("./logs", self.expr_name)
        self.writer = SummaryWriter(self.log_dir)

    def save(self, model, model_name='best_model_states.bin'):

        model_path = os.path.join(self.model_dir, model_name)
        print(f"Saving model: {model_path}", end="\n\n")
        torch.save(model.state_dict(), model_path)

    def load(self, model_path):

        print(f"Loading model: {model_path}", end="\n\n")
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def train_step(self, batch):

        self.optimizer.zero_grad()
        self.model.train()

        # get batch data
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # forward pass
        loss, logits = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False
        )

        # backward pass
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        return loss.item()

    def train_epoch(self, epoch):

        total_train_loss = 0
        i = 1
        for i, batch in enumerate(self.train_loader, 1):

            train_loss = self.train_step(batch)
            total_train_loss += train_loss

            if i % 10 == 0:
                print(
                    f"epoch: {epoch:02d}\t"
                    f"batch: {i:04d}\t"
                    f"train_loss: {train_loss:6.4f}\t"
                )

            if self.global_step % self.eval_every == 0:

                valid_loss, accuracy = self.evaluate()
                print(
                    f"\nstep {self.global_step}\t"
                    f"train_loss:{train_loss:.4f}\t"
                    f"valid_loss: {valid_loss:.4f}\t"
                    f"accuracy: {accuracy:.4f}"
                )
                self.log_evaluation(train_loss, valid_loss, accuracy)
                if self.early_stopping_counter == self.early_stopping_rounds:
                    print(f"terminating training ...")
                    break

        train_loss = total_train_loss / i
        return train_loss

    def train(self):

        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)

        for epoch in range(1, self.num_epochs+1):
            train_loss = self.train_epoch(epoch)
            if self.early_stopping_counter == self.early_stopping_rounds:
                break

        print(f"valid accuracy: {self.eval_accuracy}")

        expr_result = vars(self.args)
        expr_result['valid_accuracy'] = self.eval_accuracy

        experiments = []
        if os.path.exists("experiments.json"):
            with open("experiments.json", "r", encoding="utf-8") as f:
                experiments += json.load(f)
        experiments.append(expr_result)
        json_str = json.dumps(experiments, indent=4)

        with open("experiments.json", "w", encoding="utf-8") as f:
            f.write(json_str)

    @torch.no_grad()
    def evaluate(self):

        predictions, targets = [], []
        total_val_loss = 0.

        self.model.eval()
        for batch in self.valid_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # forward pass
            loss, logits = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=False
            )
            preds = torch.argmax(logits, dim=1)

            total_val_loss += loss.item()
            predictions += preds.tolist()
            targets += labels.tolist()

        val_loss = total_val_loss / len(self.valid_loader)
        predictions = np.array(predictions)
        targets = np.array(targets)
        accuracy = (targets == predictions).mean()

        return val_loss, accuracy

    def log_evaluation(self, train_loss, valid_loss, accuracy):

        self.writer.add_scalar("loss/train", train_loss, self.global_step)
        self.writer.add_scalar("loss/valid", valid_loss, self.global_step)
        self.writer.add_scalar("metrics/accuracy", accuracy, self.global_step)

        if accuracy > self.eval_accuracy:
            self.best_model = copy.deepcopy(self.model)
            self.eval_accuracy = accuracy
            self.save(self.model)
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            print(
                f"no performance improvement from: {self.eval_accuracy:.4f}"
                f" - early stopping [{self.early_stopping_counter}/{self.early_stopping_rounds}]",
                end="\n\n"
            )
