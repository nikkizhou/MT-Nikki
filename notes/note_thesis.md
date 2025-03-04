
Challenge: 
1. Merging datasets with synthetic data might introduce challenges in terms of ensuring the synthetic data mirrors the distribution and characteristics of the real data. It could be worth mentioning the specific methods or techniques used to ensure the synthetic data was representative and realistic.

2. design it well, in the beginning didn't think about measuring the time

3. data collection, sent many mails

4. presenting result, too many pictures


Discussion:
1. Discuss potential bias risks and misinterpretation issues when using AI:
Transformer models may learn biases from pre-trained corpora, which could affect classification fairness. To mitigate this, we performed bias analysis on model predictions.
2. Should have chosen better models for transformer-based considering the small dataset.

3. The difference in accuracy (0.93 vs. 0.91) is minor, espacially for such a small dataset. A statistical significance test could confirm whether this difference is meaningful.




TODO:
1. Go more detailed in chapter 3:
2. Train more on BiLSTM





% PEFT, LoRA
% TODO: mention Unsupervised Learning
% (time consumed, explainability(discussion)?, generalization, robustness, memeory use, parameter size,  …) 