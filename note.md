
## TODO:
1. Evaluate performance of the models
2. Print out incorrect classifications, and find why, and try to improve
3. Use Synthetic Minority Oversampling Technique (SMOTE) to artificially increase the number of instances for rare labels.  Data Augmentation strategies.
4. Analyze the results
5. TransLSTM:
https://www.sciencedirect.com/science/article/pii/S2949719124000372#sec3


## TODO if more time: 
1. Classify the answers


## Done:
1. Combine labels
2. Cross Validation
3. Find more models
4. Fine tuning:  
Learning Rate, Batch Size, Nr Epochs, Weight Decay, Dropout Rate, Gradient Accumulation Steps, Early Stopping,

5. Data Augmentation:
create diverse yet meaningful variations of questions while ensuring they still belong to the same category


Synonym Replacement: Retains structure but introduces variation.
Back Translation: Ensures meaningful variations without altering intent.
Question Reformulation: Helps create structurally diverse questions for robustness.
Paraphrasing: Generates realistic new examples close to human input.


## Questions:



## Meeting notes:

### 11.12
overfitting,
upsampling,
will get more data
category combination standard, none-question type
confusion matrix


### 11.05
    1.Models: Llama √
    2. Performance (time consumed, explainability(discussion)?, generalization, robustness, memeory use, parameter size,  …) ,
    3. Print out incorrect classifications, and find why, and try to improve


## Points to discuss in MT:

1. Data imbalance: 
   The model is heavily biased toward the option-posing class and struggles with the other classes. 
   Solution: 
   Weighted Loss for LSTM, Data Augmentation(oversampling),

2. Models: DistilBert, LLMA, Bi-LSTM
Llama: Focused on achieving high performance while using fewer resources; 

3. Overfitting with DistilBert

4. Category combination standards

5. Overfitting: 
label smoothing
Simplify the Model,
Use Data Augmentation
Use Regularization(fks. class weights, )
Apply Dropout       √
Use Early Stopping  √
Cross-Validation   √
Use Smaller Batch Sizes √
Reduced features by combining classes √
