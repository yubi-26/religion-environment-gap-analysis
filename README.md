# 🌍 Religion, Energy Structure, and Environmental Outcomes

## Can religious composition explain environmental outcomes?

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![Statistics](https://img.shields.io/badge/Statistics-Specification%20Curve-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


---

# 📌 Research Overview

This project investigates whether **religious composition predicts environmental outcomes**.

Specifically, this study examines whether:

> **Muslim population share predicts CO₂ emissions across countries after accounting for economic development, energy systems, demographic scale, and regional structures.**

The analysis uses:

- Specification Curve Analysis
- Ordinary Least Squares Regression
- Robust Standard Errors
- Regional Fixed Effects
- Machine Learning Validation

to evaluate whether the observed relationship remains consistent across alternative model specifications.

---

# 🔍 Main Finding

> **Religious composition alone does not provide a robust explanation of environmental outcomes.**

The observed association between Muslim population share and CO₂ emissions is highly sensitive to:

- Model specification
- Variable definition
- Economic structure
- Energy systems
- Regional differences


Significant associations appear in several specifications.

However:

> After controlling for regional differences, the relationship becomes statistically indistinguishable from zero.

This suggests that simple cross-national comparisons may confuse:
Religious Composition
↓
Observed Environmental Difference

with:
Economic Structure
+
Energy System
+
Regional Characteristics
↓
Environmental Outcome

---

# ❓ Research Question

## Does religious composition independently explain environmental outcomes?

Environmental outcomes differ substantially between countries.

These differences are often discussed through cultural or religious explanations.

However, countries also differ in:

- Economic development
- Energy consumption patterns
- Industrial structure
- Population size
- Geography
- Historical development


This project asks:

> Does the association between Muslim population share and CO₂ emissions remain after controlling for structural factors?

---

# 🧪 Research Design


```mermaid
flowchart TD

A[Religious Composition<br>Muslim Population Share]

A --> B[Observed Cross-National<br>CO₂ Association]

B --> C[Economic Structure<br>GDP Development]

B --> D[Energy System<br>Energy Consumption]

B --> E[Regional Factors<br>Geography & History]


C --> F[Environmental Outcome<br>CO₂ Emissions]
D --> F
E --> F
📊 Key Results
Specification Curve Analysis

Red points indicate:
p < 0.05

Gray points indicate:
p ≥ 0.05

Error bars represent 95% confidence intervals.

📈 Regression Summary
Model	Specification	Muslim Coef	p-value	R²
S1	Muslim only	-0.0019	0.313	0.007
S2	+ GDP per capita	0.0025	0.004	0.785
S3	+ GDPpc + Energy	0.0015	0.039	0.841
S4	+ GDP + Population	0.0025	0.004	0.785
S5	+ GDP + Population + Energy	0.0014	0.055	0.843
⭐ S6	+ Regional Fixed Effects	-0.0003	0.711	0.855
S7	Total CO₂ + GDPpc + Energy	0.0087	0.010	0.335
⭐ S8	Total CO₂ + GDP + Population + Energy	0.0029	0.018	0.930
⭐ S9	Absolute Muslim Population	0.0010	0.955	0.927
S10	Total CO₂ + Regional FE	0.0011	0.794	0.492
⭐ Most Important Statistical Result
Regional Fixed Effects
Model	Muslim Coefficient	p-value
GDP + Energy	0.0015	0.039
+ Regional Fixed Effects	-0.0003	0.711
Interpretation

After accounting for regional differences:

The relationship between Muslim population share and CO₂ emissions disappears statistically.

This indicates that regional and structural factors explain a substantial portion of the observed cross-country association.

However, regional controls may also absorb historically meaningful pathways connecting:

Culture
Institutions
Development patterns

Therefore, the result should be interpreted as evidence against a simple independent religious effect rather than proof that religion has no influence.

🛡️ Robustness Checks
Test	Result
Oil-exporting status	p-value: 0.039 → 0.083
Regional clustered SE	HC3 p-value: 0.055 → Cluster p-value: 0.037
Interpretation

Adding oil-exporting status reduces the estimated association.

This suggests:

Energy-related economic structures partially explain the observed relationship.

Clustered standard errors produce stronger significance, but interpretation requires caution because only six regional clusters are available.

🤖 Predictive Robustness Analysis
Integrating Econometric and Machine Learning Approaches

Machine learning models were applied as supplementary predictive tools.

The goal was not causal inference.

Instead, the analysis tested:

Does religious composition provide additional predictive information beyond structural variables?

Included predictors:

Energy consumption
Economic development
Population scale
Religious composition

Methods:

Ridge Regression

Evaluates linear relationships while reducing overfitting risk.

Random Forest Regression

Captures nonlinear relationships and interactions.

K-Fold Cross Validation

Estimates out-of-sample prediction performance.

Feature Ablation Analysis

Measures the marginal contribution of Muslim population share.

📊 Machine Learning Results
Model	Cross-Validated R²
Ridge Regression	0.8198 ± 0.034
Random Forest	0.8993 ± 0.040
Random Forest without Muslim Share	0.9014 ± 0.040
🔬 Feature Ablation Result

Removing Muslim population share:

ΔR² = -0.0021

Interpretation:

Removing religious composition slightly improved predictive performance.

This suggests that Muslim population share provides limited additional predictive information once structural variables are included.

Feature importance analysis showed:

High contribution:

Energy consumption
Economic variables

Lower contribution:

Religious composition

Because the dataset contains only 156 countries, machine learning results are interpreted as supplementary robustness evidence.

🌎 Motivation

Environmental outcomes are frequently discussed in relation to cultural and religious differences.

However, cross-national comparisons risk confusing:

Cultural composition
Economic development
Historical factors
Energy systems
Geographic structures

This project investigates whether observed relationships remain after accounting for these structural factors.

🛠️ Technical Stack
Programming
Python
Pandas
NumPy
Statsmodels
Matplotlib
Scikit-learn
Econometric Methods
Ordinary Least Squares Regression
HC3 Robust Standard Errors
Clustered Standard Errors
Regional Fixed Effects
Specification Curve Analysis
Machine Learning Methods
Ridge Regression
Random Forest Regression
K-Fold Cross Validation
Feature Ablation Analysis
Predictive Performance Comparison
📚 Data Sources
Religious Composition
Pew Research Center
Economic and Demographic Data
World Bank Development Indicators
Environmental Data
Our World in Data CO₂ Dataset
Environmental Performance Index
🔄 Research Practices

This project follows reproducible research principles:

✅ Transparent data processing pipeline

✅ Version-controlled analysis

✅ Documented variables

✅ Reproducible scripts

✅ Robustness-oriented evaluation

🚀 How to Reproduce
# Clone repository

git clone https://github.com/yubi-26/religion-environment-gap-analysis.git

cd religion-environment-gap-analysis


# Install dependencies

pip install -r requirements.txt


# Run main specification analysis

python src/05_specification_curve_analysis.py


# Run robustness checks

python src/06_robustness_checks.py
📂 Repository Structure
religion-environment-gap-analysis/

├── data/
│   ├── raw/
│   │   └── Original datasets
│   └── processed/
│       └── Cleaned datasets
│

├── src/
│   ├── 01_data_loader.py
│   ├── ...
│   ├── 05_specification_curve_analysis.py
│   └── 06_robustness_checks.py
│

├── outputs/
│   ├── tables/
│   └── figures/
│

├── paper/
│   └── paper_draft.md
│

├── research_notes.md
│

├── data_dictionary.md
│

└── README.md
⚠️ Limitations
Cross-sectional Data

The analysis uses data from a single year (2020).

Therefore:

The results identify statistical associations rather than causal effects.

Religious Measurement

Muslim population share measures demographic composition.

It does not capture:

Religious intensity
Individual beliefs
Environmental attitudes
Religious institutions
Regional Fixed Effects

Regional controls improve robustness but may absorb historical pathways connecting:

Culture
Institutions
Development
Clustered Inference

Only six regional clusters are available.

Clustered standard errors should therefore be interpreted cautiously.

🎯 Conclusion

This study examined whether Muslim population share predicts CO₂ emissions across 156 countries.

Using:

Specification Curve Analysis
Econometric Robustness Testing
Machine Learning Validation

the results show that the relationship is highly sensitive to:

Model specification
Variable definition
Regional controls

Positive associations appear in some models.

However:

These associations disappear after accounting for regional differences.

Machine learning analysis further suggests that religious composition provides limited additional predictive information beyond structural variables, especially energy-related factors.

Overall:

Environmental outcomes should not be attributed directly to religious composition without considering broader economic, demographic, energy, and institutional structures.

This project demonstrates the importance of transparency and robustness analysis when studying complex relationships between culture and environmental outcomes.

📄 License

MIT License

Copyright (c) 2026