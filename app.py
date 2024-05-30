from flask import Flask, render_template, request, jsonify
import os
import whisper
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer, AutoModel
from pydub import AudioSegment
import matplotlib
import seaborn as sns
import io
import base64
from werkzeug.utils import secure_filename
import spacy
import en_core_web_sm
import random
from textblob import TextBlob
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

matplotlib.use('Agg')  # Use the Agg backend for Matplotlib
import matplotlib.pyplot as plt

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
spacy.cli.download("en_core_web_sm")

os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

model = whisper.load_model("base")
soft_skills = ["Ethics and Integrity", "Teamwork and Collaboration", "Learning and Development", "Conscientiousness",
               "Leadership", "Communication"]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

example_responses = {
    "ethics and integrity": [
        ("Described unethical behavior or decision-making without remorse.", 1),
        ("Gave simple examples of ethical behavior, lacking depth.", 2),
        ("Demonstrated understanding of ethical principles in straightforward situations.", 3),
        ("Showed consistent ethical behavior in various challenging situations.", 4),
        ("Provided complex examples of upholding ethics and integrity under pressure.", 5)
    ],
    "teamwork and collaboration": [
        ("Described ineffective teamwork (e.g., not relying on others, unable to overcome disagreements, blaming coworkers).", 1),
        ("Demonstrated teamwork in simple or routine projects but could not give more complex examples.", 2),
        ("Helped coworkers for the good of the common goal (e.g., collaborated and shared information).", 3),
        ("Actively helped foster effective team dynamics (e.g., knowledge sharing, leveraging everyone's strengths).", 4),
        ("Described effective teamwork with complex examples.", 5)
    ],
    "learning and development": [
        ("Showed no interest in learning or improving skills.", 1),
        ("Discussed learning new skills occasionally without a clear plan.", 2),
        ("Demonstrated proactive learning in familiar situations.", 3),
        ("Consistently sought out new learning opportunities and applied new knowledge.", 4),
        ("Gave complex examples of continuous self-improvement and skill development.", 5)
    ],
    "conscientiousness": [
        ("Described careless or irresponsible behavior.", 1),
        ("Demonstrated basic responsibility in straightforward tasks.", 2),
        ("Showed reliability in routine tasks.", 3),
        ("Demonstrated high levels of reliability and attention to detail in challenging tasks.", 4),
        ("Gave complex examples of exceptional conscientiousness and thoroughness.", 5)
    ],
    "leadership": [
        ("Described poor leadership (e.g., failing to inspire, disorganization).", 1),
        ("Gave simple examples of leading small or familiar projects.", 2),
        ("Demonstrated effective leadership in familiar contexts.", 3),
        ("Showed strong leadership skills in challenging or unfamiliar situations.", 4),
        ("Provided complex examples of inspiring and effective leadership.", 5)
    ],
    "communication": [
        ("Described poor communication (e.g., misunderstandings, lack of clarity).", 1),
        ("Gave simple examples of effective communication in straightforward contexts.", 2),
        ("Demonstrated clear and effective communication in familiar situations.", 3),
        ("Showed strong communication skills in challenging or unfamiliar contexts.", 4),
        ("Provided complex examples of exceptionally clear and effective communication.", 5)
    ]
}


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalnum() and w not in stop_words]
    return ' '.join(filtered_tokens)


def standardize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [5] * len(scores)
    standardized_scores = [(10 * (score - min_score) / (max_score - min_score)) + 1 for score in scores]
    return standardized_scores


def analyze_answer(text, keywords):
    keyword_density = calculate_keyword_density(text, keywords)
    entity_count = count_named_entities(text)
    narrative_count = calculate_narrative_score(text)
    similarity_scores = calculate_cosine_similarity(text)

    keyword_density_weight = 0.2
    entity_count_weight = 0.3
    narrative_score_weight = 0.3
    similarity_score_weight = 0.2

    keyword_density_score = keyword_density * keyword_density_weight
    entity_count_score = entity_count * entity_count_weight
    narrative_score = narrative_count * narrative_score_weight
    similarity_score = np.mean(list(similarity_scores.values())) * similarity_score_weight

    total_score = keyword_density_score + entity_count_score + narrative_score + similarity_score

    if total_score == 0:
        return {
            "keyword_density": keyword_density_score,
            "entity_count": entity_count_score,
            "narrative_score": narrative_score,
            "similarity_score": similarity_score,
            "total_score": 0
        }
    else:
        return {
            "keyword_density": keyword_density_score,
            "entity_count": entity_count_score,
            "narrative_score": narrative_score,
            "similarity_score": similarity_score,
            "total_score": total_score
        }


def calculate_keyword_density(text, keywords):
    word_count = len(text.split())
    keyword_count = sum([text.lower().count(keyword.lower()) for keyword in keywords])
    keyword_density = keyword_count / word_count if word_count != 0 else 0

    repeated_keyword_penalty = sum(text.lower().count(keyword.lower()) - 1 for keyword in keywords)
    keyword_density -= repeated_keyword_penalty / word_count if word_count != 0 else 0

    return keyword_density


def count_named_entities(text):
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return len(entities)


def calculate_narrative_score(text):
    blob = TextBlob(text)
    diversity_score = len(set(blob.words)) / len(blob.words) if len(blob.words) > 0 else 0
    engaging_score = blob.sentiment.subjectivity
    narrative_score = (diversity_score + engaging_score) * 5
    return narrative_score


def generate_concept_embeddings(example_responses):
    concept_embeddings = {}
    model_name = "textattack/bert-base-uncased-ag-news"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    with torch.no_grad():
        for skill, responses in example_responses.items():
            embeddings = []
            for description, _ in responses:
                inputs = tokenizer(description, return_tensors='pt', padding=True, truncation=True)
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
            concept_embeddings[skill] = embeddings
    return concept_embeddings


def calculate_cosine_similarity(text):
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = AutoModel.from_pretrained("textattack/bert-base-uncased-ag-news")
    user_inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    user_outputs = model(**user_inputs)
    user_embedding = user_outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    similarity_scores = {}
    for skill, embeddings in concept_embeddings.items():
        cosine_similarities = []
        for example_embedding in embeddings:
            cosine_sim = cosine_similarity([user_embedding], [example_embedding])[0][0]
            cosine_similarities.append(cosine_sim)
        similarity_scores[skill] = np.mean(cosine_similarities)
    return similarity_scores


concept_embeddings = generate_concept_embeddings(example_responses)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/random-question', methods=['GET'])
def get_random_question():
    questions = [
        "Tell us about your work experiences and the skills you gained through them."
    ]
    question = random.choice(questions)
    return jsonify({'question': question})


@app.route('/record', methods=['POST'])
def record_audio():
    audio_data = request.files['audio_data']
    question = request.form.get('question', '')
    if audio_data and question:
        filename = secure_filename(audio_data.filename)
        audio_path = os.path.join('uploads', filename)
        audio_data.save(audio_path)

        if not filename.endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            audio_path = audio_path.rsplit('.', 1)[0] + '.wav'
            audio.export(audio_path, format='wav')

        result = model.transcribe(audio_path)
        cleaned_text = preprocess_text(result["text"])

        labels = [f"Good {skill.lower()}" for skill in soft_skills] + [f"Bad {skill.lower()}" for skill in soft_skills]
        classification = classifier(cleaned_text, candidate_labels=labels)
        scores = []
        for skill in soft_skills:
            good_score = classification['scores'][classification['labels'].index(f"Good {skill.lower()}")]
            bad_score = classification['scores'][classification['labels'].index(f"Bad {skill.lower()}")]
            scores.append(good_score - bad_score)

        analysis = {}
        for skill, responses in example_responses.items():
            analysis[skill] = analyze_answer(cleaned_text, keywords=[word for response in responses for word in response[0].split()])

        standardized_scores = standardize_scores(scores)
        soft_skills_scores = {skill: score for skill, score in zip(soft_skills, standardized_scores)}

        skills = list(soft_skills_scores.keys())
        scores = list(soft_skills_scores.values())

        plt.figure(figsize=(10, 5))
        sns.barplot(x=skills, y=scores)
        plt.title('Soft Skills Assessment')
        plt.xlabel('Soft Skills')
        plt.ylabel('Scores')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        return jsonify({'scores': soft_skills_scores, 'plot': plot_url, 'analysis': analysis})

    return jsonify({'error': 'No audio data or question received'}), 400


if __name__ == '__main__':
    app.run(debug=True)
