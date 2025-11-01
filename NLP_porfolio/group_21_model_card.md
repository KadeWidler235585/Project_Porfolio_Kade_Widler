# Model Card: [Emotion Classifier Transformer]

---

## 1. Model Overview

**What is your model?**
- Our model is a RoBERTa-Base transformer fine-tuned for emotion classification in text. It predicts one of 7 emotion categories (happiness, sadness, surprise, anger, disgust, fear, neutral).

**Model architecture**
- Hidden layers: 12
- Hidden size: 768
- Self-attention heads: 12
- Total parameters: around 125 million

**Design choices**
- Batch size = 16
- Attention dropout = 0.1
- Epochs = 3
- Learning rate = 2e-5

**Purpose**
*Why was this model developed?*

- This model was developed to take in cleaned transcript data from a raw video and classify it accurately as one of the [Ekman's six emotions](https://www.paulekman.com/universal-emotions/#:~:text=TO%20PAUL%20EKMAN%3F-,Dr.,seventh%20emotion%2C%20which%20is%20contempt.) or neutral. 
- This model was made to be able to preform well on a test set from the [Content Intelligence Agency Pipeline](https://www.contentintelligence.nl/). This data was extracted from [MasterChef USA (S12): Full Ep 1 | A Second Chance](https://www.youtube.com/watch?v=hUDAKXdm2BM).
- The goal of this model was to replace LLMs: reduce cost, reduce prediction time, and elimination of reliance on other companies. We did this through the implementation of a robust emotion classification pipeline. 
- This pipeline was built to processes video transcripts with high accuracy, classify emotions using multiple ML approaches, improve generalization to various content types, and provide confidence metrics for Content Intelligence Agency decision-making. 

**Development Context**

**Key Assumptions, Constraints, and Conditions**

**Key Assumptions:**
- The data from the Content Agency Pipeline was not perfect including numerous misclassifications and class imbalance. Our goal through modelling was to first develop a dataset that provided strong domain coverage for our specific test set. We did this by incorporating data from the GoEmotions Reddit data (vocabulary + slang coverage).
-  We struggled with large class imbalance in our test set: 50% happy, 25% neutral, and 25% other five emotions. Additionally, our test was small, containing 1095 sentences, a total vocab size of 3253, and 935 unique words. 

**Development conditions and constraints:**
- We experienced strong constraints in terms of training due to reliance on CPU. 
- This project was completed in eight weeks, only two of which were dedicated to modelling. With additional time, we would spend time to manually inspect the test data set, better aligning classifications.
- Training time for final model was around 10+ hours on CPU.

---

## 2. Intended Use
*How should your model be used?*

**Intended Applications:**  
- Detecting sentiment and emotional tone in video transcripts for audience analysis or improved content recommendations.

**Limitations / Misuse:**  
- Not suitable for psychoanalysis of emotions. 
- Should not be used for hiring or performance assessment.
- English and Dutch only.
- Lacks consent from individual's content.

**Client Connection:**  
- Content Intelligence Agency
- Cost Reduction: 80% reduction in manual annotation costs
- Scalability: Process 1000+ hours of content daily vs. 50 hours manually
- Consistency: Eliminate human annotator disagreement (currently 25-30% variance)
- Real-time Analysis: Enable live content moderation and recommendation

---

## 3. Dataset Details
*What data powers your model?*

**Data sources and collection methods (e.g., open datasets, proprietary data).**
- We chose the goemotions (large corpus of online reddit comments) dataset which has 234,000+ lines (from combination of all 3 csv's of GoEmotions) and around 27 emotions.

**Preprocessing steps (e.g., tokenization, normalization, filtering).**
- Emotions appropriately mapped to happiness, sadness, surprise, anger, disgust, fear, neutral
- After mapping was done, the emotions were sub-sampled to approximately 3000 per emotion leading to a final dataset size of 21000
- The emotions were label encoded:
  - "happiness": 0
  - "sadness": 1,
  - "anger": 2,
  - "surprise": 3,
  - "fear": 4,
  - "disgust": 5,
  - "neutral": 6
- Splitting and tokenization of data

***Link to our full [Data Quality Report Notebook](https://github.com/KadeWidler235585/Project_Porfolio_Kade_Widler/blob/main/NLP_porfolio/Data_quality_report.ipynb)*** 
---

## 4. Performance Metrics and Evaluation
**How does your model perform?**  

| Model                         | Dataset | Macro Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score | Train / Evaluate Time         |
|--------------------------------|----------|----------------|-----------------|---------------|----------------|-------------------|-------------------------------|
| Baseline Naive Bayes (7D)      | Test     | 0.417          | 0.610           | 0.550         | 0.420          | 0.520             | 2.1 sec                       |
| **Transformer**                    | Train    | 0.589          | 0.625           | 0.588         | 0.589          | 0.710             | 12.4 hours                    |
| Transformer                    | Test     | 0.615          | 0.510           | 0.469         | 0.475          | 0.628             | 1.2 sec                       |
| LLM                            | Test     | 0.738          | 0.680           | 0.590         | 0.573          | 0.740             | 4.4 min (55.2 min with XAI) |


- Our model currently has a Macro F1-Score of 0.589. This is a strong score for our seven classification problem. In evaluations, we tracked iterations using macro scores because the performance of all emotions mattered. 
- We built our model to preform well on the test set from the Content Intelligence Agency as well as being able to generalize well. This pipeline had errors in classifications, and our scores would be improved through manual cleaning.
- While our model had similar scores to LLM, when drilling down into class distribution, this model created detrimental strategies when classifying minority classes. For example, for the disgust class the LLM had scores of 1.00 for precision and 0.11 for recall: correctly classifying when it predicts, but missing almost all instances. This means in our test set, the LLM only predicted disgust twice out of its 18 instances.

**Transformer test set class drill-down:**

| Emotion   | Precision | Recall | F1-Score | Support |
|------------|------------|---------|-----------|----------|
| Happiness  | 0.931      | 0.674   | 0.782     | 497      |
| Sadness    | 0.691      | 0.437   | 0.535     | 87       |
| Anger      | 0.182      | 0.261   | 0.214     | 23       |
| Surprise   | 0.360      | 0.381   | 0.370     | 105      |
| Fear       | 0.571      | 0.400   | 0.471     | 70       |
| Disgust    | 0.368      | 0.389   | 0.378     | 18       |
| Neutral    | 0.468      | 0.742   | 0.574     | 295      |

- Performance on classifications has strong correlation with class size. 
---

## 5. Explainability and Transparency
**How does your model make decisions?**  

Methods used for Explainable AI (XAI) + summary of findings:
- Gradient x Input
The model was choosing words randomly, but for specific emotions, the spread of influence of words varied. For example, anger had fewer words that impacted classifications more, while in sadness, many words impacted classifications. Next
- Conservative propagation
Sentences with strong connections between words made confident classifications, while sentences with muddled connections struggled with classification. This showed us that context in making classifications for the transformer model was very important.
- Input perturbation
Emotions like neutral and sadness were much more robust emotions. When more words were removed, the confidence in these predictions remained, showing that no specific words were valuable in making these predictions. On the other hand, in emotions like fear and surprise, the removal of words had a much larger impact on confidence. This shows in making predictions for these emotions, the context of the whole sentence was needed.

***Link to our full [XAI Analysis](https://github.com/KadeWidler235585/Project_Porfolio_Kade_Widler/blob/main/NLP_porfolio/Emotion_Transformer_XAI.ipynb)*** 

---

## 6. Recommendations for Use
**Practical Guidance & Operational Risks**  
- Best practices for deployment (e.g., required preprocessing, monitoring drift).  
- Known risks or operational issues (e.g., handling out-of-distribution inputs).  
- Example use cases for **media companies or clients** — how they can integrate it into workflows.  
- Maintenance and retraining recommendations.

**Intended Use**
- The model is designed to classify emotions in sentences, the emotions it can classify are happiness, sadness, surprise, anger, disgust, fear, neutral.

**Known Risks**
- Misclassifications with subtle or figurative sentences may be misclassified
- Rare emotions like anger or disgust are harder to detect for the model

**Specific Use Cases**
- Media companie's assistance with sentiment analysis on social media and comments
- Understanding customer emotions in feedback or reviews for companies

---

## 7. Sustainability Considerations

**Environmental Impact & Mitigation**  

Estimated energy consumption and carbon footprint
- Training consumes electricity which contributes to CO2 emissions

Hardware used (e.g., GPU/CPU type, cloud provider)
- CPU mainly due to lots of GPU problems that caused laptop crashing while training 

Recommendations for reducing impact:  
- Energy-efficient hardware or data centers  
- Model compression or pruning
- Batch inference to optimize performance

---
