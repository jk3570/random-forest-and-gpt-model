from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

def predict_discrimination_type(user_message, classifier, tfidf_vectorizer):
    # Preprocess the user's message
    user_message_features = tfidf_vectorizer.transform([user_message])

    # Predict the discrimination type using the pre-trained model
    prediction = classifier.predict(user_message_features)

    # Convert the numerical prediction to a discrimination type
    discrimination_type = prediction[0]

    return discrimination_type

def get_legal_info(discrimination_type):
    legal_info_dict = {
        'gender': 'According to Republic Act 11313, also known as the Safe Spaces Act, gender discrimination is prohibited in various spheres, including employment, education, housing, and healthcare. This law protects individuals from discrimination based on their sex, gender identity, or sexual orientation.',
        'age': 'The Labor Code of the Philippines addresses age discrimination. Specifically, Article 131 prohibits discrimination against employees based on age in hiring, promotion, and termination. This ensures that individuals are judged on their qualifications and experience, not their age.',
        'disability': 'According to Republic Act 7277, discrimination based on disability is prohibited under the Magna Carta for Disabled Persons. This law guarantees equal opportunities for persons with disabilities in various aspects of life, including employment, education, and access to public spaces. It mandates reasonable accommodation for individuals with disabilities to ensure their full participation in society.',
        'religion': 'Both the Labor Code and the Magna Carta for Religious Freedom and Non-establishment of Religion address religious discrimination. The Labor Code prohibits employers from discriminating against employees based on their religion or religious beliefs, ensuring that individuals are treated fairly regardless of their faith. The Magna Carta for Religious Freedom guarantees the right to freedom of religion and protects individuals from discrimination based on their religious beliefs or practices.',
        'sexual_orientation': 'The Anti-Sexual Harassment Act of 1995 protects individuals from discrimination based on sexual orientation. This law defines sexual harassment as any unwelcome sexual advances, requests for sexual favors, and other verbal or physical conduct of a sexual nature that creates a hostile work environment. It protects individuals from such harassment regardless of their sexual orientation.',
        'race': 'The International Convention on the Elimination of All Forms of Racial Discrimination (ICERD), which the Philippines has ratified, prohibits racial discrimination. This convention requires states to take measures to eliminate all forms of racial discrimination and ensure equality before the law for all individuals, regardless of their race or ethnicity.',
        'pregnancy': 'The Expanded Maternity Leave Law protects pregnant women and new mothers from discrimination in hiring, promotion, and termination. This law mandates employers to provide paid maternity leave and other benefits to pregnant employees, ensuring their well-being and equal opportunities in the workforce.',
        'nationality': 'Both the Labor Code and the Equal Opportunities Act address nationality-based discrimination. The Labor Code prohibits discrimination against employees based on their nationality, ensuring equal treatment and opportunities for all workers. The Equal Opportunities Act prohibits discrimination in public accommodations based on national origin, ensuring fair access to services and facilities for all individuals.'
        # Add more discrimination types as needed
    }

    return legal_info_dict.get(discrimination_type, "Sorry, I couldn't find information for that discrimination type.")

def train_discrimination_model():
    # Load the dataset
    dataset = pd.read_csv('discrimination_dataset.csv')  # Acutal path of dataset

    # Separate features (user messages) and target variable
    features = dataset['user_message']
    target = dataset['discrimination_type']

    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the features (text data) using TF-IDF
    features_transformed = tfidf_vectorizer.fit_transform(features)

    # Split the dataset into training and testing sets
    features_train, features_test, target_train, target_test = train_test_split(
        features_transformed, target, test_size=0.2, random_state=42
    )

    # Initialize the RandomForestClassifier
    classifier = RandomForestClassifier()

    # Train the classifier
    classifier.fit(features_train, target_train)

    # Load pre-trained GPT model and tokenizer
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    return {'classifier': classifier, 'tfidf_vectorizer': tfidf_vectorizer,
            'gpt_model': gpt_model, 'gpt_tokenizer': gpt_tokenizer, 'dataset': dataset}
