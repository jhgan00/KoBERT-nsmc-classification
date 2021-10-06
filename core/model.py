from transformers import BertForSequenceClassification


def get_classifier(bert_model_name):
    model = BertForSequenceClassification.from_pretrained(
        bert_model_name,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels: 2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    return model
