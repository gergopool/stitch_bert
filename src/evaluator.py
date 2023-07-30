import torch


def evaluate(model, data_loader):

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set model trainable
    trainable_was = model.training
    model.eval().to(device)

    preds, labels = [], []
    for batch in data_loader:

        # Prepare input
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[2],
            "attention_mask": batch[1],
            "labels": batch[3]
        }

        # Calculate loss
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**inputs)

        # Get predictions
        preds.append(outputs[1].cpu().detach())
        labels.append(batch[3].cpu().detach())

    # Concatenate predictions and labels
    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Set model back to trainable
    model.train(trainable_was)

    return preds, labels
