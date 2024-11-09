
## TODO:
1. Evaluate performance of the models
2. Print out incorrect classifications, and find why, and try to improve
3. Use Synthetic Minority Oversampling Technique (SMOTE) to artificially increase the number of instances for rare labels.  Data Augmentation strategies.
4. Analyze the results


## TODO if more time: 
1. Classify the answers


## Done:
1. Combine labels
2. Cross Validation
3. Find more models
4. Fine tuning:  Learning Rate, Batch Size, Nr Epochs, Weight Decay, Dropout Rate, Gradient Accumulation Steps, Early Stopping,


## Questions:


## Meeting notes:
### 11.05
    1.Models: Llama √
    2. Performance (time consumed, explainability?, generalization, robustness, memeory use, parameter size,  …)
    3. Print out incorrect classifications, and find why, and try to improve


## Notes

1. Data imbalance: 
   The model is heavily biased toward the option-posing class and struggles with the other classes. 
   Solution: 
   Weighted Loss for LSTM, Data Augmentation(oversampling),

2. Models: DistilBert, LLMA, Bi-LSTM
Llama: Focused on achieving high performance while using fewer resources; 


