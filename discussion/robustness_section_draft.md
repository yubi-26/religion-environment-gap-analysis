# Robustness Analysis

## 1. Why robustness analysis is necessary

The initial analysis examined whether Muslim population share is associated with CO₂ emissions across countries. However, cross-country environmental data are vulnerable to confounding because countries differ substantially in economic structure, energy systems, geographic location, and historical development.

Therefore, statistical significance in a single regression specification cannot establish that religious composition itself explains environmental outcomes.

To evaluate whether the observed relationship is robust, multiple alternative specifications and analytical approaches were tested.


## 2. Robustness checks conducted

This study conducted five main robustness checks:

### (1) Specification curve analysis

Multiple regression specifications were compared by changing:

- control variables
- fixed effects structure
- measurement methods of religious variables

The purpose was to examine whether the Muslim coefficient remained stable across reasonable modeling choices.

**Key finding**: The coefficient changed from positive to negative across specifications, and statistical significance varied. This instability suggests that the relationship is not robust.

### (2) Oil-exporter control

Because several countries with high Muslim population shares are major fossil fuel exporters, an oil-exporter dummy variable was added.

**Key finding**: The Muslim coefficient changed from statistically significant (p = 0.039) to non-significant (p = 0.083) after adding oil-exporter status. This indicates that part of the observed association may reflect fossil-fuel-based economic structures rather than religious composition itself.

### (3) Regional fixed effects

Regional fixed effects were introduced to control for broad geographic and historical differences.

**Key finding**: After controlling for regions, the Muslim coefficient became statistically unsupported (p = 0.711). This suggests that regional structures explain a substantial portion of the observed cross-country relationship.

### (4) Clustered standard errors

Because countries within the same region may not be statistically independent, region-clustered standard errors were tested.

**Key finding**: The p-value changed from 0.055 (HC3) to 0.037 (cluster). However, because only six region clusters were available, this result should be interpreted cautiously. The primary inference remains based on HC3 robust standard errors.

### (5) Machine learning auxiliary analysis

A limited machine learning analysis was conducted using Ridge regression and Random Forest with five-fold cross-validation.

The objective was not prediction optimization, but evaluating whether religious composition provides additional predictive information beyond structural variables.

**Key finding**: 
- Energy consumption was the dominant predictive factor
- Removing Muslim share slightly improved Random Forest performance:
  - Full model: R² = 0.8993
  - Without Muslims: R² = 0.9014
  - ΔR² = -0.0021

Therefore, under this model specification, religious composition did not provide additional predictive improvement.


## 3. Summary of findings

Across different analytical approaches, the relationship between Muslim population share and CO₂ emissions was unstable.

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| Specification curve | Coefficients varied in sign and significance | Relationship is specification-dependent |
| Oil-exporter control | p: 0.039 → 0.083 | Oil structure partially explains the association |
| Regional FE | p = 0.711 | Regional structure explains the association |
| Clustered SE | p: 0.055 → 0.037 | Inference is sensitive to error structure |
| ML Feature Ablation | ΔR² = -0.0021 | Religion does not improve prediction |

The results do not support a simple interpretation that religious composition directly determines environmental outcomes.


## 4. Most supported mechanism

Among competing explanations, the **energy structure hypothesis** receives the strongest support.

Countries with high CO₂ emissions are primarily characterized by:

- high energy consumption
- fossil fuel dependence
- industrial structure

rather than religious composition itself.

This conclusion is supported by:
1. ML feature importance showing energy as the dominant predictor
2. Feature ablation showing religion does not improve model performance
3. The instability of the Muslim coefficient across structural controls

However, this study does not claim that energy structure completely explains all variation, because energy systems are themselves influenced by economic development, institutions, and historical factors.


## 5. Limitations and future research

Several limitations remain.

**First**, the analysis is primarily cross-sectional, limiting causal interpretation. The observed associations do not establish that energy consumption "causes" CO₂ emissions, nor that religious composition has no causal effect.

**Second**, Muslim population share captures demographic composition rather than religious beliefs, practices, or environmental ethics. This study cannot speak to whether individual religious values influence environmental behavior.

**Third**, energy variables may themselves be endogenous because they are influenced by economic and institutional factors. The observed dominance of energy in the ML model may reflect this endogeneity rather than a pure structural effect.

**Fourth**, the number of region clusters (6) is small, so clustered standard error results should be interpreted cautiously.

**Future research** should incorporate:

- longer panel datasets to examine temporal dynamics
- institutional indicators (governance, policy, legal systems)
- direct measurements of religious practices and environmental attitudes
- spatial dependence analysis
- more comprehensive energy structure variables (renewable share, efficiency metrics)


## 6. Conclusion

The main contribution of this study is not demonstrating a direct religious effect on environmental outcomes.

Instead, it shows that apparent relationships between cultural categories and environmental indicators can disappear when structural factors are carefully considered.

**The key findings are:**

1. The Muslim-CO₂ association is sensitive to model specification
2. Regional and oil-related structures explain much of the observed relationship
3. Energy consumption is the dominant predictor in ML models
4. Religious composition does not improve predictive performance

**The study's contribution** is not "religion has no effect," but rather:

> Simple cross-national comparisons can mistake structural differences for cultural effects. Environmental outcomes are primarily shaped by economic, technological, and institutional structures rather than religious composition alone.

This highlights the importance of moving beyond simple cultural explanations and analyzing the structural mechanisms underlying environmental outcomes.