# Religious Composition and Environmental Outcomes
## A Specification Curve Analysis of Muslim Population Share and CO₂ Emissions

---

## Abstract

This study examines the relationship between Muslim population share and CO₂ emissions using cross-national data from 156 countries. Through specification curve analysis across 10 model specifications, we find that the association is highly sensitive to model choice. While significant positive associations appear in models controlling for GDP and energy (p < 0.05), these associations disappear when regional fixed effects are introduced (p = 0.711). Additionally, measurement choice matters: Muslim population share is significant (p = 0.018), but Muslim population size is not (p = 0.955). These findings suggest that observed relationships are largely explained by regional economic and demographic structures rather than a direct religious effect.

---

## 1. Introduction

[簡単な背景]
- 環境問題は世界的な課題
- 宗教と環境の関係についての議論がある
- しかし、単純な相関ではなく交絡要因を考慮する必要がある

### Research Question
> Does Muslim population share predict CO₂ emissions after controlling for economic, demographic, and regional factors?

---

## 2. Data and Methods

### 2.1 Data
- 156 countries, 2020
- Sources: World Bank, Pew Research Center, EPI

### 2.2 Variables
- **Dependent**: log(CO₂ per capita), log(Total CO₂)
- **Independent**: Muslim population share (%)
- **Controls**: log(GDP per capita), log(GDP), log(Population), Energy consumption
- **Fixed Effects**: Region

### 2.3 Specifications (S1-S10)

| Model | Dependent | Muslim Var | Controls | Purpose |
|-------|-----------|------------|----------|---------|
| S1 | CO2_pc | Share | None | Baseline |
| S2 | CO2_pc | Share | GDPpc | Economic control |
| S3 | CO2_pc | Share | GDPpc + Energy | Energy control |
| S4 | CO2_pc | Share | GDP_total + Pop | Scale control |
| S5 | CO2_pc | Share | GDP_total + Pop + Energy | Full controls |
| **S6** | **CO2_pc** | **Share** | **GDPpc + Energy + Region FE** | **Regional structure** |
| S7 | Total CO2 | Share | GDPpc + Energy | Total emissions |
| **S8** | **Total CO2** | **Share** | **GDP_total + Pop + Energy** | **Full scale control** |
| **S9** | **Total CO2** | **Pop count** | **GDP_total + Pop + Energy** | **Measurement test** |
| S10 | Total CO2 | Share | GDPpc + Energy + Region FE | Regional + total |

---

## 3. Results

### 3.1 Specification Curve Analysis

**Figure 1** shows the Muslim coefficient across all 10 specifications.

Key observations:
- Coefficients range from **-0.0019 to +0.0087**
- 5/10 specifications are statistically significant (p < 0.05)
- 5/10 are not significant

### 3.2 Regional Fixed Effects

**Critical finding**: Adding region fixed effects eliminates significance.

| Specification | Muslim Coef | p-value | R² |
|---------------|-------------|---------|-----|
| S3 (GDPpc + Energy) | 0.0015 | **0.039** | 0.841 |
| **S6 (+ Region FE)** | **-0.0003** | **0.711** | **0.855** |

→ **Region FE explains the association.**

### 3.3 Measurement Matters

| Specification | Variable | p-value |
|---------------|----------|---------|
| S8 | Muslim **share** | **0.018** ✅ |
| S9 | Muslim **population** | **0.955** ❌ |

→ **How you measure Muslim population changes the conclusion.**

### 3.4 Oil-Exporting Status

| Model | Muslim p-value |
|-------|----------------|
| Without oil | 0.039 |
| With oil | 0.083 |

→ Oil partially explains, but does not fully explain.

### 3.5 Clustered Standard Errors

| SE Type | p-value |
|---------|---------|
| HC3 | 0.055 |
| Cluster (region) | 0.037 |

→ Region clusters affect inference (caution: only 6 clusters).

---

## 4. Discussion

### 4.1 Interpretation

The relationship between Muslim population share and CO₂ emissions is **not robust across specifications**.

- Without regional controls: significant
- With regional controls: **not significant**

This suggests that **regional economic and energy structures** (oil economies, development patterns) explain the observed relationship, rather than religious composition itself.

### 4.2 Why This Matters

This study demonstrates:
1. **Specification curve analysis** is essential for robustness
2. **Region fixed effects** absorb substantial variation
3. **Measurement choice** (share vs count) changes inference

### 4.3 Limitations

- Cross-sectional data → no causal inference
- Muslim share is not religious intensity
- Only 6 region clusters for clustered SE
- Limited to 156 countries

---

## 5. Conclusion

**Main finding:**

> The association between Muslim population share and CO₂ emissions is highly specification-dependent. Significant associations appear in some models but disappear when regional fixed effects are introduced. This suggests that regional economic and demographic structures, not religious composition per se, explain the observed relationship.

**Contribution:**

> This study provides empirical evidence that cultural-environmental relationships must be examined with careful attention to model specification, measurement, and regional structure.

---

## References

[To be completed]

---

## Tables and Figures

### Table 1: Summary of Results

| Model | Muslim Coef | p-value | R² | Significant? |
|-------|-------------|---------|-----|--------------|
| S1 | -0.0019 | 0.313 | 0.007 | ❌ |
| S2 | 0.0025 | 0.004 | 0.785 | ✅ |
| S3 | 0.0015 | 0.039 | 0.841 | ✅ |
| S4 | 0.0025 | 0.004 | 0.785 | ✅ |
| S5 | 0.0014 | 0.055 | 0.843 | ❌ |
| **S6** | **-0.0003** | **0.711** | **0.855** | **❌** |
| S7 | 0.0087 | 0.010 | 0.335 | ✅ |
| **S8** | **0.0029** | **0.018** | **0.930** | **✅** |
| **S9** | **0.0010** | **0.955** | **0.927** | **❌** |
| S10 | 0.0011 | 0.794 | 0.492 | ❌ |

### Figure 1: Specification Curve
[Placeholder for image]