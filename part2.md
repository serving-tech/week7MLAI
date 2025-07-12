# Part 2: Case Study Analysis

## Case 1: Biased Hiring Tool – Amazon’s AI Recruiting System

### Scenario
Amazon developed an AI-powered recruiting tool trained on 10 years of résumé data. Because most applicants in the tech industry were male, the AI system learned to favor male candidates and penalize female-associated terms such as “women’s chess club” or degrees from all-women’s colleges.

### Source of Bias
The bias stemmed from historical data reflecting gender imbalance in the tech industry. This data encoded societal biases, which were then learned and perpetuated by the AI model. Even with efforts to remove gendered terms, subtle proxies continued to influence the model.

### Proposed Fixes to Improve Fairness

1. **Balanced and Representative Training Data**
   - Augment the training data with résumés from qualified women.
   - Use oversampling or synthetic generation to ensure balanced demographic representation.
   - Eliminate gender-proxy variables through better feature selection and pre-processing.

2. **In-Processing and Post-Processing Mitigation**
   - Apply fairness-aware algorithms during training (in-processing).
   - Adjust decision thresholds after training to reduce group-level disparities (post-processing).

3. **Human Oversight and Audits**
   - Keep humans in the loop during decision-making.
   - Regularly audit the model using fairness metrics to detect emerging biases.

### Fairness Metrics

- **Disparate Impact**: Ensures selection rate for women is at least 80% of the rate for men.
- **Equal Opportunity Difference**: Measures the difference in true positive rates across demographic groups.
- **Conditional Demographic Disparity**: Checks whether identical scores correlate to different selection probabilities across groups.

---

## Case 2: Facial Recognition in Policing

### Scenario
Facial recognition systems, like Amazon's Rekognition, misidentify people of color—especially Black women—at significantly higher rates, leading to wrongful arrests and systemic harms.

### Ethical Risks

1. **Wrongful Arrests and False Accusations**
   - False matches lead to arrests of innocent individuals.
   - Disproportionate impact on Black communities.

2. **Surveillance and Privacy Violations**
   - Pervasive public surveillance chills free expression and violates privacy rights.

3. **Amplification of Systemic Bias**
   - Use in minority communities increases biased data and enforcement cycles.

4. **Opacity and Due Process Failures**
   - Proprietary systems are black boxes; defendants can’t challenge their reliability in court.

### Policy Recommendations

- **Ban Real-Time Mass Surveillance**: Restrict use to specific, warrant-backed investigations.
- **Independent Testing**: Require demographic-specific accuracy evaluation before deployment.
- **Mandatory Human Review**: Treat facial recognition as an investigative lead, not arrest evidence.
- **Public Transparency**: Agencies must disclose usage policies, usage logs, and oversight mechanisms.
- **Protect Due Process**: Guarantee disclosure of algorithmic use in legal proceedings, including access to error rates and system metadata.
