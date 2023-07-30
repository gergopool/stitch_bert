from transformers import AutoModelForSequenceClassification, glue_processors


def build_pretrained_transformer(model_type: str, task_name: str):
    n_classes = len(glue_processors[task_name]().get_labels())
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=n_classes)
    return model