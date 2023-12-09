import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Assuming you have ground truth labels and predicted labels as tensors
ground_truth = torch.tensor([0, 1, 1, 0, 1, 0, 1, 1])
predicted_labels = torch.tensor([0, 1, 0, 0, 1, 1, 1, 1])

# Convert tensors to numpy arrays
ground_truth_np = ground_truth.numpy()
predicted_labels_np = predicted_labels.numpy()

# Create confusion matrix
conf_matrix = confusion_matrix(ground_truth_np, predicted_labels_np)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = accuracy_score(ground_truth_np, predicted_labels_np)
print(f"Accuracy: {accuracy:.4f}")

# Calculate precision
precision = precision_score(ground_truth_np, predicted_labels_np)
print(f"Precision: {precision:.4f}")

# Calculate recall
recall = recall_score(ground_truth_np, predicted_labels_np)
print(f"Recall: {recall:.4f}")

# Calculate F1 score
f1 = f1_score(ground_truth_np, predicted_labels_np)
print(f"F1 Score: {f1:.4f}")

