import numpy as np
import torch
import treform as ptm
import csv


class Predictor:

    def __init__(self, model, tokenizer, pipeline=None, max_len=100, device="cpu"):

        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.max_len = max_len
        self.device = device
        self.model.to(self.device)

    def preprocess(self, documents):

        if self.pipeline is not None:
            corpus = ptm.Corpus(textList=documents)
            documents = self.pipeline.processCorpus(corpus)
            documents = [" ".join(word for sent in doc for word in sent) for doc in documents]

        return documents

    def tokenize(self, documents):

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids, attention_masks, token_type_ids = [], [], []

        # For every sentence...
        for document in documents:
            encoded_dict = self.tokenizer.encode_plus(
                document,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=self.max_len,  # Pad & truncate all sentences.
                padding="max_length",
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            token_type_ids.append(encoded_dict['token_type_ids'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        return batch

    @torch.no_grad()
    def classify(self, batch):

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)

        (logits,) = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        return preds, probs

    def predict(self, documents):
        documents = self.preprocess(documents)
        inputs = self.tokenize(documents)
        preds = self.classify(inputs)
        return preds

    def test(self, loader):

        documents, predictions, labels, negatives, positives = [], [], [], [], []

        self.model.eval()
        for batch in loader:

            preds, probs = self.classify(batch)
            predictions += preds.tolist()
            negs, pos = zip(*probs.tolist())
            documents += batch['document_text']
            labels += batch['labels'].tolist()
            negatives += negs
            positives += pos

        predictions = np.array(predictions)
        labels = np.array(labels)
        accuracy = (labels == predictions).mean()

        with open("predictions.txt", "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(['documents', 'prediction', 'label', 'negative', 'positive'])
            for row in zip(documents, predictions, labels, negatives, positives):
                writer.writerow(row)

        print(f"test accuracy: {accuracy:.4f}")
        return accuracy

    def interact(self):

        while True:

            comment = input("comment >>> ")
            if comment.lower() == "exit":
                break
            prediction, probability = self.predict([comment])
            probability = probability.to("cpu").numpy().round(4).squeeze() * 100
            print("negative:", probability[0])
            print("positive:", probability[1])
