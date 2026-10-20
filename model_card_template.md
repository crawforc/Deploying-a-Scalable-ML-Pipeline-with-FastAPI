# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

I built this model for the Udacity Machine Learning DevOps course. It uses a Random Forest classifier from scikit-learn. I set random_state to 42 so I get the same results every time I run it. The model predicts if a person earns more than $50K a year. Before training, the text columns were changed into numbers so the model could use them.

## Intended Use

This model is for learning only. It shows how to train a model, test it, and run it through an API.

## Training Data

The data is the Census Income dataset from the UCI Machine Learning Repository. It comes from the 1994 U.S. Census. It has 32,561 rows and 15 columns. I used 80% of the data to train the model, which is 26,048 rows.

## Evaluation Data

I used the other 20% of the data to test the model, which is 6,513 rows. The model did not see this data during training. This shows how well the model works on new data.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

I used three metrics to check the model: precision, recall, and F1 score.

- Precision shows how often the model is right when it predicts someone earns more than $50K.
- Recall shows how many of the people who really earn more than $50K the model found.
- F1 score combines precision and recall into one number.

My model's results on the test data:

- Precision: 0.7419
- Recall: 0.6384
- F1 score: 0.6863

When the model says someone earns more than $50K, it is right about 74% of the time. It finds about 64% of the people who really earn more than $50K, so it misses about 36% of them.

I also checked the model on smaller groups in the data, such as each education level. Those results are saved in slice_output.txt.

## Ethical Considerations

The data includes race, sex, and country of origin. A model trained on this data could be biased based on the selection of the data provided. The results in slice_output.txt show the model does better for some groups than others. Because of this, the model should not be used for decisions that affect people's lives.

## Caveats and Recommendations

The data is from 1994, so it is out of date. The $50K cutoff is not adjusted for today's income levels. Some groups have very few people in the test data, so their results may not be representative. Missing values show up as "?" and were left in the data. To improve the model, I would use newer data, try different model settings, and look closer at the groups where the model does poorly.