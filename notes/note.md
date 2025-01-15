
## TODO:
1. Evaluate performance of the models
2. Print out incorrect classifications, and find why, and try to improve
5. TransLSTM:
https://www.sciencedirect.com/science/article/pii/S2949719124000372#sec3


## TODO if more time: 


## Done:
1. Combine labels
2. Cross Validation
3. Find more models
4. Fine tuning:  
Learning Rate, Batch Size, Nr Epochs, Weight Decay, Dropout Rate, Gradient Accumulation Steps, Early Stopping,
5. Combine more datas
6. Fix overfitting problem
7. Data Augmentation:
create diverse yet meaningful variations of questions while ensuring they still belong to the same category

Synonym Replacement: Retains structure but introduces variation.
Back Translation: Ensures meaningful variations without altering intent.
Question Reformulation: Helps create structurally diverse questions for robustness.
Paraphrasing: Generates realistic new examples close to human input.


## Questions:
1. In my cross validation process, I evaluate the model every fold the the real world data in that fold. After that, is it good practice to evaluate the model whole real world dataset?
Then we don't need to generate confusion matrix for every fold, only the one with whole real world data.

## Meeting notes:

### 01.07:
Start thesis:
implementation and design part, add the result to the thesis
find papers, research


### 12.10:
Other latest models like hunyuan? Hailuo ?
Bi-LSTM, confusion matrix, and analyze
draw graghs: fks before and after class weigh

Consider changing to title to 'Improving Model Accuracy with Limited Training Data in the context of CC'
Improving Model Accuracy  in the context of CC interviews data
does it improve? why not?

### 11.17


### 11.12
upsampling,
confusion matrix


### 11.05
    1.Models: Llama √
    2. Performance (time consumed, explainability(discussion)?, generalization, robustness, memeory use, parameter size,  …) ,
    3. Print out incorrect classifications, and find why, and try to improve


## Points to discuss in MT:

### Challenges:
1. The first file and the last two file has different format. Technical challenge to merge them into same df
2. Upsampling, technical challenge
3. load the dataset, for cross validation, need to mark the synthetic data; For not CV, need to add the synthetic data to original data then split them to train and validation.
4. When add synthetic data to the dataset, Only when both ADD_SYNTHETIC_DATA and USING_CROSS_VALIDATION are true,   'is_synthetic' will be marked. But it should be sufficient with only ADD_SYNTHETIC_DATA then mark  'is_synthetic'.  The problem is caused by not thinking through in the beginning. and the code are added after.

### 1. Data imbalance: 
   The model is heavily biased toward the option-posing class and struggles with the other classes. 
   Solution: 
   Weighted Loss for LSTM, Data Augmentation(oversampling),
   class_weights = compute_class_weights

### 2. Models: DistilBert, LLMA, Bi-LSTM
Llama: Focused on achieving high performance while using fewer resources; 

### 3. Overfitting with DistilBert, but not other models.


### 4. Using k-fold cross validation: the 4th and 5th fold start overfitting, so I adjusted to 3 fold
Stratified KFold (ensures class balance in each fold)

### 5. Overfitting: 
label smoothing
Simplify the Model,
Use Data Augmentation
Use Regularization(fks. class weights, )
Apply Dropout       √
Use Early Stopping  √
Cross-Validation   √
Use Smaller Batch Sizes √
Reduced features by combining classes √
Use Synthetic Minority Oversampling Technique (SMOTE) to artificially increase the number of instances for rare labels. √



### 6. Category combination standards:
Challenge with combining categories:
Data are in different format.
file 1 has multi label problem
Difficult to find a unified standard for combination standard.

suggetive: passively
leading: explicitly, leading try both (tag & statement) combined and seperately

#### File 1:  Categorized_mocks.xlsx
1. Original Label Counts, SUM: 2715
R2_3YN    676    
R2_3      646
R2_5      527
R2_2B     326
R2_2D     222
R2_2SD     87
R2-1       71
R2_OP      62
R2_4QL     40
R2_4QG     27
R2_4QR     19
R2_4QV      8
R2_4QP      2
R2_4QI      1
R2_6        1

2. Combined based on original combination standard:
| Label | Category         | Count | Origianl labels
|-------|------------------|-------|-------------------------
|   0   | invitation       | 69    | ['R2-1']
|   1   | directive        | 537   | ['R2_2B', 'R2_2D', 'R2_2SD']
|   2   | option-posing    | 1050  | ['R2_3', 'R2_3YN', 'R2_OP']
|   3   | suggestive       | 81    | ['R2_4QG', 'R2_4QL', 'R2_4QP', 'R2_4QR', 'R2_4QI', 'R2_4QV']
|   4   | none-questions   | 927   | ['R2_5']
|   5   | multiple         | 51    | ['R2_6']


3. Combine invitation and directive into open-ended. 
And drop suggestive and multiple, cz it's too little samples:
Label 0 (open-ended): 706
Label 1 (option-posing): 1384
Label 2 (none-questions): 544
Label 3 (suggestive): 80
Label 4 (multiple): 1

#### File 2: Question Type examples 9_20_24.xlsx
1. Original Label Counts:
Open-ended             337
DYK                    206
option posing          206
option-posing            7
option posiing           1
leading (tag)          171
leading (statement)    140

2. Combined Label Counts:
Merge 'DYK/DYR' into 'option-posing'
open-ended       337
option-posing    420
leading          311


#### File 3: Forensic Trafficking Interviews Question Type Examples 10_1_24.xlsx
1. Original Label Counts:
open-ended                     347
option-posing                  289
DYK/DYR                        200
Leading , tag                  110
leading, statement question    154
leading (statement)              2

2. Combined Label Counts:
Merge 'DYK/DYR' into 'option-posing'
open-ended       347
option-posing    489
leading          266


#### All Three Files Combined, SUM: 4804:
open-ended        1390
option-posing     2293
none-questions     544
leading            577



NEW!
| Label | Main Category  | Categories Combined to main Category| File 1 | File 2 | File 3 | Total  |
|-------|----------------|-------------------------------------|--------|--------|--------|--------|
| 0     | open-ended     | invitation,directive                | 706    |  337   |  347   |  1390  |
| 1     | option-posing  | DYK/DYR                             | 1384   |  420   |  489   |  2293  |
| 2     | none-questions |                                     | 544    |  0     |  0     |  544   |
| 3     | leading        | leading (tag), leading (statement)  | 0      |  311   |  266   |  577   |


#### Label Count 3 files + synthetic Data: 

Synthetic data should only be in training, test data shouldn't synthesic data
In cross validation, synthesic data is excluded in validation set. Therefore, the validation set size varies in each fold.

1. split origianl 80% and 20% for test
2. test it
3. then add syn data 80% and 20% for test

same number of samples for each categories

