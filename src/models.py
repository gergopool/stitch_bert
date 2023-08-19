from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, glue_processors


def build_pretrained_transformer(model_type: str, task_name: str):
    if task_name == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(model_type)
    else:
        n_classes = len(glue_processors[task_name]().get_labels())
        model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=n_classes)
    return model
