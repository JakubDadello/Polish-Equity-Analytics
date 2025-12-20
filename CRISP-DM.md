# Description of the project
The analysis of a company's financial data, taking into account the characteristics of its sector, is among the key factors considered in investment-related decision-making processes. The analytical project presented in this document focuses on collecting selected, critical, and structured financial and non-financial data concerning Polish publicly listed companies. Subsequently, a comprehensive process of data exploration and visualization was carried out.

---

# Purpose of the project
The purpose of this project was to develop an efficient classification model capable of accurately categorizing the investment assessment of companies based on the mentioned determinants.

---

# Project issues
1. How can the investment evaluation of a publicly listed company be conducted based on selected financial indicators and information regarding its sector affiliation?
2. What type of analytical model, based on machine learning algorithms, ensures sufficient effectiveness in Solving this Business problem?

---

# Business Understanding
This project concerns fundamental analysis, understood within the framework of economic sciences and financial market theory as a set of methods used to evaluate the value of a company’s financial assets, particularly equity instruments.
Fundamental analysis is based on the company’s so-called “fundamentals,” closely linked to the concept of “intrinsic value.” Serving cognitive, descriptive, explanatory, and predictive functions, it provides a valuable repertoire of analytical techniques and tools for entities investing capital in financial markets.

Due to the multidimensional and interdisciplinary nature of fundamental analysis, the project was focused on one of its core stages — financial analysis.
Financial analysis is the process of examining a company’s financial condition in order to determine its investment attractiveness and the efficiency of its business operations. It is based on the assessment of selected financial ratios, with data sourced from the financial statements of the companies under analysis.

Financial Analysis Areas: 
1. Balance sheet analysis (assets and liabilities)
2. Profitability analysis (income statement/profit and loss account)
3. Cash flow analysis (cash flow statement)
4. Liquidity analysis (e.g., current ratio)
5. Profitability ratios analysis (e.g., ROE)
6. Debt/leverage analysis (e.g., long-term debt ratio)

---

# Data Understanding
The Data Understanding process was carried out for each type of data. In this project, these included both financial and non-financial data.
The tools used during the Data Understanding phase included the PostgreSQL database management system and the pgAdmin interface. In the next step, the preprocessed data, enriched with investment evaluation labels, were imported into an Excel file. Power BI was then used for their visualization.

Within this project, for each company listed on the Warsaw Stock Exchange (WSE), numerical data were collected illustrating the values of five fundamental financial indicators for the year 2025:
1. Net profit (net income/loss)
2. Net cash flow
3. Return on equity (ROE)
4. Return on assets (ROA)
5. Operating profit plus depreciation (EBITDA)

The selection of the above indicators was driven by the need to cover all areas of financial analysis that are highly relevant. They encompass the profitability, liquidity, and efficiency of the company under analysis.

The primary source of the collected financial data was the individual financial statements of the analyzed companies, as included in their periodic reports.
Depending on formal legal requirements and the companies’ internal reporting and accounting policies, for some companies the analyzed reports covered the first quarter of 2025. For the remaining companies, the examined period encompassed the first half of the year, so the statements were of a cumulative nature.

The non-financial data collected for the purposes of this project are categorical in nature and cover thirteen sectors:
1. agriculture and food industry;
2. chemicals and health products;
3. commercial services;
4. communication technologies and media;
5. consumer goods and e-commerce;
6. energy technology;
7. financial and insurance;
8. health and life sciences;
9. hospitality industry;
10. manufacturing industry;
11. real estate and construction;
12. technology and engineering;
13. transport and motorization.

Each analyzed company was assigned to one of the aforementioned sectors. The main criterion for selection was the company’s business profile, primarily including its product structure and raw material–capital base. For companies with a highly diversified range of products or services, categorization was based on the dominant product portfolio. The official Warsaw Stock Exchange (WSE) website served as the source of data regarding each company’s sector affiliation.

more information about dataset in `data/README.md`

---

# Data Preparation 

Data Preparation Steps Conducted in the Project:
1. Exporting the Excel file containing the preprocessed data.
2. Separating the label column (investment rating) from the rest of the data.
3. Splitting the comprehensive dataset into numerical and categorical data.
4. Creating a numerical transformer in the form of a sequential processing pipeline, used for preprocessing numerical data, including imputing missing values and normalizing random variables through standardization.
5. Creating a categorical transformer, also in the form of a Pipeline, intended for preprocessing categorical data, including imputation and conversion of categorical variables into numerical format using OneHotEncoding.
6. Creating a combined object that incorporates both the numerical and categorical transformers.

The entire data preparation process was implemented in the file preprocessing.py.

---

# Modeling
The objective of this project was to create a classification model that assigns an investment rating to each company, reflecting its financial condition and taking into account its sector affiliation.

To achieve this goal, four popular classification models based on machine learning algorithms were used:
1. Logistic Regression
2. Support Vector Machine (SVM), specifically the classification variant (SVC)
3. Random Forest, applied to classification problems
4. Artificial Neural Network (ANN), precisely a multilayer perceptron classification model (MLP Classifier)

The source code for each model is contained in a separate file within the notebooks folder.

For building the Logistic Regression, SVM, and Random Forest models, the corresponding tools from the Python library scikit-learn were used. In contrast, the neural network was created using functions from the TensorFlow library, implemented via the specialized deep learning interface Keras.
The training process involved estimating the model parameters (weights) with the goal of minimizing the loss function. Each model was trained six times. After each training cycle, predictions were generated, based on which three evaluation metrics were calculated: accuracy, precision, and recall, along with the confusion matrix.

Each new training cycle of the model was preceded by adjustments to its hyperparameters for optimization.
1. For Logistic Regression, the C hyperparameter, which controls the strength of regularization, was modified, and an additional operation was implemented during the Data Preparation phase - dimensionality reduction using Principal Component Analysis (PCA).
2. The SVM was optimized primarily with respect to the kernel function (linear, polynomial, Gaussian) and the gamma hyperparameter.
3. In the case of the Random Forest, changes included the number of decision trees and the splitting criterion (Gini, entropy).
4. For the perceptron-based neural network, the number of layers and neurons was optimized. During the last two cycles, the dropout technique was implemented to reduce the risk of overfitting.

---

# Evaluation 

The previously mentioned evaluation process for each of the four models involved calculating model predictions after each of the six training cycles, and then deriving one of the three main evaluation metrics based on these predictions:
1. Accuracy
2. Precision
3. Recall

Each model was accompanied by an Excel table containing the metric values for every cycle.
The evaluation process was further enhanced with a confusion matrix.

| Logistic Regression |
|-----------------------------------|
| accuracy | precision | recall |
|----------|-----------|-----------|
| 0.4594594594594595 | 0.44078947368421| 0.434813886568272 |


Logistic Regression results:
1. Accuracy ranged from 46% to 50%, indicating that the model correctly classified about half of the cases.
2. Precision remained relatively stable at approximately 45%.
3. Recall was within a narrow range of 43–46%.
In the last (sixth) cycle, all metrics reached their highest values, yet none exceeded 50%.
Overall, the Logistic Regression model did not achieve satisfactory performance, mainly due to the lack of linear separability between the features and the probabilistic logits.

SVM results:
1. Accuracy hovered around 43%, showing no significant fluctuations.
2. Precision increased by almost 20 percentage points in the fifth cycle, reaching 57%, and then dropped to around 50% in the final trial.
3. Recall remained within a narrow range, oscillating at approximately 40%.
In general, the SVM model performed slightly worse than the previously tested Logistic Regression model in solving this business problem. It should be noted that, besides the linear variant (LinearSVC), nonlinear kernels (polynomial and Gaussian) were also tested.
The limited effectiveness of SVM was likely due to features lacking proper scaling. Differences in magnitude—such as net profit in millions versus profitability ratios in percentages—significantly influenced the creation of the decision hyperplane and, consequently, the classification quality.

Random Forest results:
1. Accuracy ranged between 75% and 80%.
2. Precision remained in a similar range, dropping below 75% only in the fifth cycle.
3. Recall behaved similarly to precision throughout the testing process, with comparable values in each cycle.
Random Forest clearly outperformed all other tested machine learning models.
Its success in solving the classification problem is likely due to its structure. By hierarchically dividing the multidimensional feature space into smaller decision trees, the model can capture nonlinear and complex feature interactions.

Neural Network results:
Accuracy remained relatively stable, with the exception of the fourth cycle, when changing the optimizer caused it to drop to around 36%.
Other metrics also declined noticeably after switching from the Adam optimizer to SGD.
Overall, the multilayer perceptron (MLP) neural network developed for this project showed performance comparable to Logistic Regression and SVM, despite having  significantly more complex internal architecture and operational characteristics.

---

# Deployment 

The analytical model developed in this project, based on the CRISP-DM methodology, is of a prototype nature and is not yet suitable for full deployment.
It requires a number of further actions, primarily related to:
1. Modifying its structure, including increasing the number of numerical and categorical attributes. This would allow the model to take into account a larger set of factors determining its attractiveness for potential capital investments.
2. Expanding the dataset by including more companies, which would enhance the system’s applicability to international financial markets.
3. Functionally, it would be valuable to explore additional machine learning algorithms for classification tasks. These could include bagging-based ensemble classifiers, such as Random Forest, which performed best with the current project structure.
4. Within ensemble methods, boosting algorithms—including adaptive boosting (AdaBoost) and gradient boosting (XGBoost) - are also worth testing. This recommendation applies to the current dataset.
4. With an increase in model structural complexity, deep learning approaches, such as the previously applied perceptron neural network, might achieve even greater effectiveness.

The project aimed to create a classification system that automatically evaluates a company’s investment attractiveness. This objective was largely met.
The evaluation method involved first grouping companies based on the temporal aspect of their financial statements and then by sector membership. Deterministic class assignment was based on a fundamental statistical measure of central tendency—the median of the relevant financial indicators.
Currently, the Random Forest algorithm demonstrates the highest performance, pointing the way for further development and optimization of the analytical model.



Analysis of the confusion matrix suggests that the model is currently moderately stable and suitable for preliminary classification. Nevertheless, its ability to accurately predict companies with medium investment attractiveness is limited, indicating a preference for the extreme classes (high and low investment ratings).
For further development of the Random Forest model, the key areas for optimization include hyperparameter tuning, feature engineering, and rebalancing of class distributions.
