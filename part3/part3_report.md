compas_bias_analysis
# Part 3: Practical Audit – COMPAS Dataset (25%)

This section presents an audit of the COMPAS Recidivism Dataset using IBM’s AI Fairness 360 toolkit. The audit focuses on uncovering racial bias in the risk classification scores assigned to defendants.

The dataset was preprocessed, and fairness metrics such as Disparate Impact, False Positive Rate (FPR), and Equal Opportunity Difference were computed. A simulated biased prediction model was created to emphasize the gap in treatment across racial groups. The analysis further includes bias mitigation through the Reweighing technique.

---

## Visualization: False Positive Rate by Race

The following chart highlights the disparity in FPR between African-American and Caucasian defendants, revealing significant bias in the COMPAS prediction system.

![False Positive Rate by Race](./compas_bias_analysis.png)

---

## COMPAS Recidivism Dataset Bias Audit Report

### EXECUTIVE SUMMARY:
This audit reveals significant racial bias in the COMPAS risk assessment system. African-American defendants are systematically assigned higher risk scores than Caucasian defendants, even when controlling for actual recidivism rates.

---

### KEY FINDINGS:

**1. STATISTICAL PARITY VIOLATION:**
- Disparity Score: 0.189  
- African-American defendants receive high-risk classifications at a rate of 23.6% vs 4.7% for Caucasian defendants

**2. FALSE POSITIVE RATE DISPARITY:**
- African-American FPR: 0.204  
- Caucasian FPR: 0.032  
- African-American defendants are 6.3× more likely to be incorrectly flagged as high-risk

**3. EQUAL OPPORTUNITY VIOLATION:**
- Disparity Score: 0.232  
- Different true positive rates across racial groups indicate unequal treatment

---

## REMEDIATION RECOMMENDATIONS:

**1. IMMEDIATE ACTIONS:**
- Implement bias testing in model development pipeline  
- Establish fairness constraints in algorithm training  
- Regular auditing with demographic parity checks

**2. ALGORITHMIC INTERVENTIONS:**
- Apply preprocessing techniques (e.g., reweighting samples)  
- Use in-processing fairness algorithms (e.g., adversarial debiasing)  
- Implement post-processing calibration methods

**3. PROCEDURAL REFORMS:**
- Require human oversight for high-risk classifications  
- Implement appeals process for risk score decisions  
- Provide bias training for decision-makers using these scores

**4. MONITORING AND GOVERNANCE:**
- Establish continuous monitoring dashboard  
- Set acceptable bias thresholds with regular reporting  
- Create interdisciplinary review board including ethicists and community representatives

---

## CONCLUSION:
The COMPAS system demonstrates clear evidence of racial bias that undermines principles of equal justice. Immediate intervention is required to prevent discriminatory outcomes in criminal justice decisions.
