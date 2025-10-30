# Model Card: [Group 21 Transformer]

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

- This model was developed to take in cleaned transcript data from a raw video and classify it accurately as one of the [[Ekman's six emotions](https://www.paulekman.com/universal-emotions/#:~:text=TO%20PAUL%20EKMAN%3F-,Dr.,seventh%20emotion%2C%20which%20is%20contempt.)] or neutral. 
- This model was made to be able to preform well on a test set from the [[Content Intelligence Agency Pipeline](https://www.contentintelligence.nl/)]. This data was extracted from [[MasterChef USA (S12): Full Ep 1 | A Second Chance](https://www.youtube.com/watch?v=hUDAKXdm2BM)].
- The goal of this model was to replace LLMs: reduce cost, reduce prediction time, and elimination of reliance on other companies. We did this through the implementation of a robust emotion classification pipeline. 
- This pipeline was built to processes video transcripts with high accuracy, classify emotions using multiple ML approaches, improve generalization to various content types, and provide confidence metrics for Content Intelligence Agency decision-making. 

**Development Context**

**Key Assumptions, Constraints, and Conditions**

**Key Assumptions:**
- The data from the Content Agency Pipeline was not perfect including numerous misclassifications and class imbalance. Our goal through modelling was to first develop a dataset that provided strong domain coverage for our specific test set. We did this by incorporating data from the GoEmotions Reddit data (vocabulary coverage).
-  We struggled with large class imbalance in our test set: 50% happy, 25% neutral, and 25% other five emotions. Additionally, our test was small, containing 1095 sentences, a total vocab size of 3253, and 935 unique words. 

**Development conditions and constraints:**
- I had bad constraints in terms of training since we had to mainly rely on slow CPU training for the transformer model and I tried to implement GPU with cuda but it wouldn't work properly frequently on my laptop since it would crash and restart it while training and I would have to restart.
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
- English only.
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
- We chose the goemotions (large corpus of online reddit comments) dataset which has 234,000+ lines (from combination of all 3 csvs of goemotions) and around 27 emotions.

**Preprocessing steps (e.g., tokenization, normalization, filtering).**
- Emotions appropriately mapped to happiness, sadness, surprise, anger, disgust, fear, neutral
- After mapping was done, the emotions were sub-sampled to approximately 3000 per emotion leading to a final dataset size of around 21000
- The emotions were label encoded into labels through 0-6
  - "happiness": 0
  - "sadness": 1,
  - "anger": 2,
  - "surprise": 3,
  - "fear": 4,
  - "disgust": 5,
  - "neutral": 6
- Splitting and tokenization of data

**Dataset composition (e.g., size, class distribution, languages covered).**
- Final dataset size was around 21000
- Test data was the output of the Content Intelligence Agency company pipeline on our chosen MasterChef episode with emotions. It had a large class imbalance in our test set: 50% happy, 25% neutral, and 25% other five emotions. Additionally, our test was small, containing 1095 sentences, a total vocab size of 3253, and 935 unique words. 
- Emotions happiness, sadness, surprise, anger, disgust, fear, neutral were distributed around 3000 per class after sub-sampling
- Language covered English only.

**Representativeness across cultures and demographics**
- It involves quite a few cultures online however more skewed towards western and english speaking countries, they can under-represent non-western countries and non-english speaking countries

---

## 4. Performance Metrics and Evaluation
**How does your model perform?**  

We addressed this topic by performing a thorough error analysis of the models mistakes, by looking at the confusion matrix, as well as the classification report we managed to gather a lot of insights. We also dove deeper into specific examples on what mistakes the model made with its worst performing emotion as well as specific examples with two emotions it confused with each other. We also discussed which metrics would be the most important to us as well as look at the strengths and weaknesses overall of the model.

***Link to our full [error analysis PDF](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-21-1/blob/main/Deliverables/ILO7.4/Error_analysis.pdf)***

---

## 5. Explainability and Transparency
**How does your model make decisions?**  

We addressed this topic by using three different XAI techniques to get a better understanding of how the transformer model thinks, since its basically a black box and we want to attempt to understand it better.

Methods used for Explainable AI (XAI):
- Gradient x Input
- Conservative propagation
- Input perturbation

We looked through each method and wrote down our findings by looking at specific examples, where we picked three examples from each emotion and analyzed them.

***Link to our full [XAI findings PDF](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-21-1/blob/main/Deliverables/ILO3.5/XAI_findings_group_21.pdf)***

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
- Media companies assistance with sentiment analysis on social media and comments
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
