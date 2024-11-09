## Fine tuning: Key Differences Between Transformer and Non-Transformer Models

Transformer Models:
1. Learning rate warm-up and scheduler are often used.
2. You typically freeze earlier layers in the transformer model.
3. Gradient accumulation is more frequently used due to memory constraints with large models.
4. Layer normalization and position encodings are unique to transformers and not applicable to most non-transformer models.
5. Pre-trained models are fine-tuned on task-specific data, so fewer epochs are often sufficient.


Non-Transformer Models:
1. Non-transformer models (e.g., CNNs, RNNs, MLPs) generally do not require learning rate warm-up, though it can still be useful.
2. Regularization techniques like L2 regularization (weight decay) or dropout are often the primary ways to avoid overfitting.
3. Gradient clipping is often used in training models like RNNs to avoid exploding gradients.
4. Hyperparameters like learning rate or optimizer settings may have a larger impact due to the more complex learning dynamics when working with traditional neural networks.