# ML-Summative
# SOBER: Controlling Drug Addiction in the African/African American Creative Industry

## Abstract

Drug addiction poses a significant threat to the health and productivity of creatives in the African/African American creative industry. This project aims to develop a machine learning model that predicts addiction risk based on individual behavioral patterns and provides a support system with tailored tips. By increasing awareness and providing targeted interventions, this model aims to reduce addiction rates and improve the well-being of creatives in this vulnerable population.  This project leverages data-driven insights to address the specific challenges of this community, contributing to a healthier and more productive creative industry.

## 1. Introduction and Motivation

The African creative industry is a vibrant hub of talent, yet its creatives face a disproportionately high rate of addiction challenges.  Statistics show alarming trends, with opioid-related deaths among African/African Americans increasing significantly (Smart, 2022).  Limited access to mental health services (WHO) and a rising substance use rate (WHO, 2022) further exacerbate this issue. This project, "SOBER," seeks to address this critical need by leveraging machine learning to predict addiction risk and provide personalized support. By analyzing behavioral patterns and offering targeted interventions, SOBER aims to reduce addiction and enhance the well-being of creatives.

## 2. Problem Statement

Drug addiction is a complex issue with devastating consequences, and the African/African American creative industry is particularly vulnerable.  Factors such as high-stress environments, limited access to mental health resources, and cultural influences contribute to this increased risk.  This project focuses on developing a data-driven solution to identify individuals at risk and provide them with the necessary support to overcome addiction.

## 3. Dataset

The dataset used in this project is sourced from UCI/Kaggle and contains information about drug consumption for 1885 individuals. It includes features such as:

* Age
* Gender
* Education Level
* Country of Residence
* Ethnicity
* Personality Traits (Neuroticism, Extraversion, Openness, Agreeableness, Conscientiousness)
* Impulsivity
* Sensation Seeking
* Self-Reported Drug Usage for Various Substances (Alcohol, Amphetamines, Cannabis, etc.)

The target variable represents the level of drug usage, categorized into seven distinct classes.  The "Semer" column, which contains only NaN values, was removed during preprocessing.  Categorical features were encoded using Label Encoding.  Missing values were imputed using the mean strategy.  The data was then split into training, validation, and test sets (80%, 10%, 10% respectively), and numerical features were scaled using StandardScaler.

## 4. Model Architecture

The model architecture consists of a neural network with the following layers:

* Input Layer: Matches the number of features in the dataset.
* Hidden Layer 1: 64 neurons, ReLU activation.
* Hidden Layer 2: 32 neurons, ReLU activation.
* Output Layer: 7 neurons, Softmax activation (for 7 addiction risk categories).

[Include Model Architecture Image Here]

## 5. Training and Evaluation

The model was trained using the following configurations:

* **Loss Function:** Sparse Categorical Crossentropy (appropriate for multi-class classification with integer labels).
* **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.
* **Optimizers:** Adam (default and tuned), RMSprop, SGD.
* **Regularization:** L2 regularization and Dropout were explored.
* **Epochs:** 50-100 (varied by model).
* **Batch Size:** 32-64 (varied by model).

The training process involved feeding the training data to the model, calculating the loss and gradients, and updating the model's weights using the chosen optimizer. The validation data was used to monitor the model's performance during training and prevent overfitting. The test data was used for the final evaluation of the trained model.

## 6. Results

The performance of the models varied depending on the chosen optimizer and regularization techniques. The best performing model (Model 2: Adam with L2 regularization and Dropout) achieved the following results on the test set:

* Accuracy: [Insert Accuracy Value]
* Precision: [Insert Precision Value]
* Recall: [Insert Recall Value]
* F1-score: [Insert F1-score Value]

[Include Confusion Matrix Plot Here]

The loss curves for the trained models are shown below.

[Include Loss Curve Plots for all models Here]


## 7. Prediction

To make predictions on new data using the trained model, the following steps are required:

1.  **Load the saved model:** Use `keras.models.load_model("path/to/your/model.h5")`.
2.  **Preprocess the new data:** Ensure that the new data is in the same format as the training data and that the numerical features are scaled using the same `StandardScaler` that was fit on the training data.
3.  **Make predictions:** Use the loaded model's `predict()` method to generate predictions. The output will be probabilities for each of the seven classes. Use `np.argmax()` to get the predicted class labels.

## 8. Usage

To run this project, you will need to have Python and the required libraries (TensorFlow, Keras, scikit-learn, pandas, numpy, matplotlib, seaborn) installed.  Then, follow these steps:

1.  Clone the repository.
2.  Install the necessary dependencies: `pip install -r requirements.txt` (create a `requirements.txt` file listing all the libraries used).
3.  Run the Jupyter Notebook.
4.  Follow the instructions in the notebook to train the models and make predictions.

## 9. Conclusion and Future Work

This project demonstrates the potential of machine learning to predict addiction risk in the African/African American creative industry.  The trained models can be used to identify individuals at risk and provide them with timely interventions.  Future work could focus on:

* Gathering more data to improve the model's accuracy and generalizability.
* Exploring different model architectures and hyperparameters.
* Developing a user-friendly interface for the model.
* Incorporating user engagement metrics to assess the effectiveness of the interventions.
* Collaborating with healthcare providers to integrate the model into existing support systems.
* Deploying the model as a mobile app or web service.

## 10. Key Metrics (From Proposal)

* User Engagement (active users, session duration, frequency of use).
* Addiction Level Reduction (pre- and post-usage assessment).
* Revenue (subscriptions/partnerships).

## 11. Unique Value Proposition (From Proposal)

* Integrates behavioral analysis with region-specific data and cultural understanding.
* Provides a comprehensive, data-driven approach to addiction management tailored to each user.

# Results of Optimizations and Parameter Setting

| Training Instance | Optimizer Used | Regularizer Used | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1-Score | Recall | Precision | Loss |
| :------------------ | :------------- | :--------------- | :----- | :------------- | :----- | :------------ | :------- | :------- | :----- | :-------- | :--- |
| Instance 1          | Adam (Default) | None             | 50     | No             | 2      | 0.001 (Default) | 0.306    | 0.285    | 0.306  | 0.300     | 1.838 |
| Instance 2          | Adam           | L2 (0.001)       | 50     | No             | 2      | 0.001         | 0.311    | 0.291    | 0.311  | 0.304     | 1.834 |
| Instance 3          | RMSprop        | None             | 100    | No             | 2      | 0.0001        | 0.306    | 0.283    | 0.306  | 0.298     | 1.835 |
| Instance 4          | Adam           | None             | 50     | No             | 2      | 0.001         | 0.306    | 0.285    | 0.306  | 0.300     | 1.838 |
| Instance 5          | Adam           | None             | 50     | No             | 2      | 0.001         | 0.306    | 0.285    | 0.306  | 0.300     | 1.838 |

## 12. Unfair Advantage (From Proposal)
[Link to Proposal](https://docs.google.com/document/d/1UrOqmwfIvFnwYE6D0rup4zAvXc0G_DpJrOXjlUPmnbY/edit?tab=t.0#heading=h.gem7vt95w6br)
(https://docs.google.com/document/d/1UrOqmwfIvFnwYE6D0rup4zAvXc0G_DpJrOXjlUPmnbY/edit?tab=t.0#heading=h.gem7vt95w6br)
* Solid partnerships and affiliations with healthcare, agricultural, and wellness providers.

[Link to Video]((https://www.loom.com/share/b19ddc2bdbf04f118118baa8cef4d77e?sid=212cbf06-71d1-4118-bf6f-4179a1bdb393))
https://www.loom.com/share/b19ddc2bdbf04f118118baa8cef4d77e?sid=212cbf06-71d1-4118-bf6f-4179a1bdb393

[Link to Neural Network DIagram](https://drive.google.com/file/d/1JtiVIzMV9JTRVbrd3Grg3e-GWwZX6VaL/view?usp=sharing)
https://drive.google.com/file/d/1JtiVIzMV9JTRVbrd3Grg3e-GWwZX6VaL/view?usp=sharing
