# app.py
from flask import Flask, render_template, request, jsonify
from discrimination_model import predict_discrimination_type, train_discrimination_model, get_legal_info
from collections import deque

app = Flask(__name__)

# Load the model and tokenizer when the application starts
discrimination_model = train_discrimination_model()

# Unpack the components from the discrimination_model dictionary
classifier = discrimination_model['classifier']
tfidf_vectorizer = discrimination_model['tfidf_vectorizer']
gpt_model = discrimination_model['gpt_model']
gpt_tokenizer = discrimination_model['gpt_tokenizer']
dataset = discrimination_model['dataset']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['user_message']
    bot_response, discrimination_type = get_bot_response(user_message)
    return jsonify({'response': bot_response, 'discrimination_type': discrimination_type})

def generate_human_like_response(user_message):
    input_ids = gpt_tokenizer.encode(user_message, return_tensors="pt")
    output = gpt_model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    return gpt_tokenizer.decode(output[0], skip_special_tokens=True)

def get_bot_response(user_message):
    # Call the Random Forest Classifier to predict discrimination type
    discrimination_type = predict_discrimination_type(user_message, classifier, tfidf_vectorizer)

    # Define legal_info outside of the if-else block
    legal_info = ""

    # Provide specific responses based on user input
    if "hi" in user_message.lower() or "hello" in user_message.lower():
        bot_response = "Hello Leonard, how can I help you today?"
    elif "how are you" in user_message.lower():
        bot_response = "I'm just a bot, but thanks for asking!"
    elif "what is your name?" in user_message.lower():
        bot_response = "My name is Gab."
    elif "can you help me?" in user_message.lower():
        bot_response = "Yes, I'm here to help you! What's your problem?"
    elif "are you gabai?" in user_message.lower():
        bot_response = "Yes, I'm Gab"
    elif "i feel discriminated today" in user_message.lower():
        bot_response = "I'm sorry to hear that, but I'm here to help you."
    elif "do you want to listen to me?" in user_message.lower():
        bot_response = "Yes, I'm here to listen to you."
    elif "thank you" in user_message.lower():
        bot_response = "You're welcome!"
        
    else:
        # Check if the user message is in the discrimination dataset
        if user_message.lower() in dataset['user_message'].str.lower().values:
            # Provide legal information based on discrimination type
            legal_info = get_legal_info(discrimination_type)
            bot_response = legal_info
        else:
            # Generate a more human-like response using Hugging Face GPT
            bot_response = generate_human_like_response(user_message)

    return bot_response, discrimination_type


if __name__ == '__main__':
    app.run(debug=True)
