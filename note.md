## Problems:
1. Data imbalance


## Ideas:
1. Fine tune
2. Find more models

1. Consider up-sampling or down-sampling to balance the dataset, or even exploring data augmentation strategies.
2. Stratified cross-validation
3. Use Synthetic Minority Oversampling Technique (SMOTE) to artificially increase the number of instances for rare labels. This can help balance the dataset for training and testing


## Done:
1. Combine labels
3. Cross Validation


## Question:
1. In file result_BERT.out:
Unique actual labels: {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13}
Expected labels based on label_columns: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} 