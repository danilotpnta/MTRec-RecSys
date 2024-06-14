import torch
import torch.nn.functional as F

def calculate_accuracy(logits, one_hot_targets):
    """
    Calculates the accuracy for each segment of the logits array with one-hot encoded targets.

    Args:
        logits (torch.Tensor): Flattened array of logits.
        repeats (torch.Tensor): Tensor indicating the number of logits in each segment.
        one_hot_targets (torch.Tensor): Flattened one-hot encoded target labels.

    Returns:
        float: The accuracy of the predictions.
    """
    #assert logits.ndim == 1, "Logits should be a flattened array"
    #assert one_hot_targets.ndim == 1, "One-hot targets should be a flattened array"

    # Split logits and one-hot targets according to repeats
    #split_logits = torch.split(logits, repeats.tolist())
    #split_targets = torch.split(one_hot_targets, repeats.tolist())
    
    # Determine the maximum length for padding
    #max_len = max(repeats)
    
    # Pad logits and one-hot targets, then stack them
    #padded_logits = torch.stack([F.pad(segment, (0, max_len - len(segment)), 'constant', float('-inf')) for segment in split_logits])
    #padded_targets = torch.stack([F.pad(segment, (0, max_len - len(segment)), 'constant', 0) for segment in split_targets])
    
    # Apply softmax to logits and get the predicted class
    softmaxed_logits = F.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(softmaxed_logits, dim=-1)
    
    # Get true class indices from one-hot targets
    true_classes = torch.argmax(one_hot_targets, dim=-1)
    
    # Calculate accuracy
    correct_predictions = (predicted_classes == true_classes).float()
    accuracy = correct_predictions.mean()
    
    return accuracy

def f1_score(y_true, y_pred):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return 2 * (precision * recall) / (precision + recall + 1e-8)