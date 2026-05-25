# MH6822 Assignment 1

ZHANG JUNYI
G2505266E
JUNYI006@e.ntu.edu.sg
# MH6822 Option A Assignment 1

## Submission Files

# MH6822 Option A Assignment 1

## Submission Files

### 🎥 Task 4 Presentation Video

Please watch the presentation video below:

https://github.com/user-attachments/assets/94777bd1-01c0-455c-8286-053c3f6f5090

If the video does not preview, open it directly here:  
[Open Task 4 Presentation Video](https://github.com/user-attachments/assets/94777bd1-01c0-455c-8286-053c3f6f5090)

### 🎧 Task 4 MP3 Audio File

The MP3 audio file is available here:  
[Open / Download Task4_Presentation_ZHANG_JUNYI_G2505266.mp3](把你刚生成的MP3链接放这里)
## Topic

A Jurisdiction-Aware AI Credit Governance Tool for Cross-Border Banks: Comparing US Fair Lending Explainability and EU AI Act Bias Governance.

## Regulated entity and domain

The selected regulated entity is Citi / Citigroup. The selected domain is AI credit scoring, fair lending, algorithmic fairness and AI governance. The primary comparison is between the United States and the European Union.

Singapore and UK appear in the jurisdiction rule file only as extensible design examples. The substantive assignment analysis focuses on US and EU.

#Recommended reading order for Task 3:

1. Task3_Tool_Design_Deck.pdf
   Main senior-management presentation of the jurisdiction-aware RegTech tool.

2. Task3_One_Page_Summary.pdf
   Plain-language summary of the design choices and key result.

3. Task3_AI_Credit_Governance_Prototype_executed.ipynb
   Executed working prototype showing data generation, model training, fairness metrics,
   jurisdiction-specific outputs and sensitivity analysis.

4. Task3_Model_Card.pdf
   Governance stub documenting model purpose, intended users, assumptions, limitations
   and failure modes.

5. Task3_Technical_Appendix.pdf
   Technical details on synthetic data, metric formulas, jurisdiction logic and outputs.

6. Task3_Sensitivity_Analysis.pdf and Task3_SHAP_Explainability.pdf
   Supporting analyses for threshold sensitivity and applicant-level explainability.
   
## Contents

- `Task1_Selection_and_Research.pdf`: entity selection, domain rationale, jurisdictional comparison and references.
- `Task2_Values_Audit.pdf`: mission statement, stakeholder perspective, risk-vs-documentation discussion and error-cost analysis.
- `Task3_Tool_Design_Deck.pdf`: senior management deck.
- `Task3_Dashboard_Mockup.pdf`: visual dashboard mockup of approval rates, thresholds, adverse-action reasons and sensitivity.
- `Task3_SHAP_Explainability.pdf`: applicant-level SHAP-style explanation plots and mapping to adverse-action reason categories.
- `Task3_AI_Credit_Governance_Prototype_executed.ipynb`: pre-executed notebook with saved cell outputs.
- `Task3_Model_Card.pdf`: governance documentation stub and model card.
- `Task3_Model_Comparison.pdf`: logistic-regression primary model versus Random Forest challenger benchmark.
- `Task3_One_Page_Summary.pdf`: plain-language one-page summary.
- `Task3_Sensitivity_Analysis.pdf`: quantitative sensitivity analysis of threshold conclusions under protected-group score shifts.
- `Task3_Technical_Appendix.pdf`: technical appendix explaining data generation, model logic, formulas, jurisdiction logic and limitations.
- `Task3_build_assignment.py`: generation script for data, PDFs, notebook and archive.
- `data/Task3_synthetic_credit_applications.csv`: simulated application-level data.
- `data/Task3_jurisdiction_rules.csv`: configurable jurisdiction rule layer.
- `data/Task3_model_outputs.csv`: model outputs and jurisdiction-specific reporting text.
- `data/Task3_fairness_metrics_by_jurisdiction.csv`: fairness metrics and governance actions by active jurisdiction.
- `data/Task3_sensitivity_analysis.csv`: sensitivity test results for US and EU thresholds.
- `data/Task3_shap_explanations.csv`: top applicant-level SHAP-style feature contributions for rejected applicants.
- `data/Task3_model_comparison.csv`: primary versus challenger model metrics.
- `data/Task3_random_forest_feature_importance.csv`: Random Forest challenger feature importance table.
- `data/Task3_eu_ai_act_obligations.csv`: EU AI Act obligation-to-control matrix for Article 10, Article 14 and Article 17.
- `data/Task3_data_dictionary.csv`: field-level data dictionary, including model-use status, protected/proxy status, synthetic-bias role and production-use cautions.

## Data note

This assignment uses synthetic credit application data. The fields are designed with reference to public lending and credit-risk datasets and regulatory concepts, but the data is not real Citi customer data and should not be interpreted as evidence about Citi's actual models, customers or compliance posture.

The `model_score` column is generated after training the primary logistic-regression model. It is the trained model's predicted probability of default, not a separate pre-model synthetic score.

## Data collaboration

The synthetic data for this prepared submission was generated independently. No other students were involved in the data generation portion.

## How to reproduce the generated artifacts

Run:

```bash
python3 Task3_build_assignment.py
```

In the working folder used to prepare the submission, the same script is named `build_assignment.py`.
