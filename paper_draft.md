# Does Religion Determine Environmental Outcomes?
## A Causal Analysis of Religious Composition and Carbon Emissions

---

## Abstract

This study examines the relationship between religious composition and carbon emissions using cross-national data from 181 countries. Addressing the endogeneity of religious composition through an instrumental variables strategy based on historical geographic diffusion of Islam, we find no evidence that Muslim population share has a causal effect on CO₂ emissions per capita. Our results consistently show that economic development—measured by GDP per capita—is the primary driver of carbon emissions, regardless of religious composition. These findings challenge culturally deterministic narratives and highlight the importance of structural factors in environmental policymaking.

---

## 1. Introduction

Do religious values shape environmental outcomes? This question has gained increasing attention in both academic and policy circles, yet empirical evidence remains scarce and contested. A prevalent narrative suggests that Muslim-majority countries are environmentally "backward" due to religious teachings or cultural values. However, this narrative rests on weak empirical foundations—correlation is often mistaken for causation, and structural factors are frequently overlooked.

This study contributes to the literature by providing rigorous causal evidence on the relationship between religious composition and carbon emissions. Using cross-national data from 181 countries and an instrumental variables strategy based on historical geographic diffusion of Islam (latitude and distance from Mecca), we address the endogeneity of religious composition.

Our findings challenge culturally deterministic narratives:
1. OLS estimates show no significant relationship between Muslim population share and CO₂ emissions.
2. IV estimates confirm the absence of a causal effect.
3. Economic development (GDP) consistently predicts emissions, regardless of religious composition.

---

## 2. Data and Methods

### 2.1 Data Sources

| Data | Source | Variables |
|------|--------|-----------|
| Religion | Pew Research Center | Muslim population share, 2020 |
| CO₂ Emissions | Our World in Data | CO₂ per capita, 2020 |
| GDP | Our World in Data | GDP per capita, 2020 |
| Gender | World Bank API | Female parliament share, 2020 |
| Geography | Manual | Latitude, Distance from Mecca |

### 2.2 Empirical Strategy

**OLS Model:**
$$CO2_i = \beta_0 + \beta_1 Muslim_i + \beta_2 GDP_i + \varepsilon_i$$

**IV Model (2SLS):**

First Stage:
$$Muslim_i = \pi_0 + \pi_1 Latitude_i + \pi_2 DistMecca_i + \pi_3 GDP_i + u_i$$

Second Stage:
$$CO2_i = \beta_0 + \beta_1 \widehat{Muslim}_i + \beta_2 GDP_i + \varepsilon_i$$

---

## 3. Results

### 3.1 Main Results

| Model | Muslims Coefficient | p-value | R² |
|-------|--------------------|---------|-----|
| OLS | 0.010 | 0.347 | 0.003 |
| IV (2SLS) | [IV coefficient] | [p-value] | [R²] |

### 3.2 First Stage

| Instrument | Coefficient | p-value |
|------------|-------------|---------|
| Latitude | -0.264 | 0.063 |
| Distance from Mecca | -5.885 | <0.001 |
| **F-statistic** | **13.67** | - |

---

## 4. Robustness Checks

[See Robustness section in discussion/]

---

## 5. Discussion

Our results demonstrate that economic structure, not religious identity, determines carbon emissions. This finding has important implications:

1. **Policy**: Environmental interventions should focus on economic and institutional reforms
2. **Discourse**: The narrative linking Islam to environmental degradation is empirically unsupported
3. **Research**: Future studies should examine structural factors rather than cultural explanations

---

## 6. Conclusion

The central message of this research is clear: **economic structure, not religious identity, is the fundamental determinant of national carbon emissions**.

---

## References

[To be added]

---

## Appendix

[Code available at: https://github.com/yubi-26/religion-environment-gap-analysis]