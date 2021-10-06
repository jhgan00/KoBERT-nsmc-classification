import torch
import csv
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


class PYBERTDataset(Dataset):

    def __init__(self, contents, targets, tokenizer, max_len):

        super(PYBERTDataset, self).__init__()
        self.contents = contents
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, item):
        content = str(self.contents[item])
        target = self.targets[item]

        encoding = self.tokenizer(
            content,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'document_text': content,
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(
        filename,
        tokenizer,
        max_len,
        batch_size,
        num_workers=4,
        shuffle=False,
        doc_colname="document",
        target_colname="label",
        delilmiter="\t"
    ):

    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delilmiter)
        headers = next(reader, None)
        data = defaultdict(list)

        for row in reader:
            for h, v in zip(headers, row):
                data[h].append(v)
        data[target_colname] = [int(y) for y in data[target_colname]]

    ds = PYBERTDataset(
        contents=data[doc_colname],
        targets=data[target_colname],
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
