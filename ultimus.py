import streamlit as st
import nltk
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Configuration & Cache
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    return nltk.corpus.stopwords.words('french')

STOP_WORDS = setup_nltk()

# 2. Logique du Chatbot optimisée
def chatbot_response(user_input, sentences, vectorizer, tfidf_matrix):
    # Transformer uniquement l'entrée utilisateur
    query_vec = vectorizer.transform([user_input.lower()])
    
    # Calcul de similarité
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = similarities.argsort()[-1]
    highest_score = similarities[best_idx]
    
    if highest_score < 0.1:  # Seuil de pertinence
        return "Désolé, je ne trouve pas de réponse pertinente dans le fichier."
    
    return sentences[best_idx]

# 3. Fonction de reconnaissance vocale
def transcribe_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("Microphone activé, je vous écoute...", icon="🎤")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            return r.recognize_google(audio, language="fr-FR")
        except sr.UnknownValueError:
            return "Erreur : Je n'ai pas compris l'audio."
        except sr.RequestError:
            return "Erreur : Problème de connexion au service vocal."
        except Exception as e:
            return f"Erreur : {e}"

# 4. Interface Streamlit
st.set_page_config(page_title="Chatbot Pro", page_icon="🤖")
st.title("🤖 Chatbot Intelligent (Texte & Voix)")

uploaded_file = st.file_uploader("Chargez votre corpus (.txt)", type="txt")

if uploaded_file:
    # Lecture et préparation (une seule fois)
    raw_text = uploaded_file.read().decode("utf-8")
    sentences = nltk.sent_tokenize(raw_text)
    
    # Vectorisation du corpus (Optimisé)
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    option = st.sidebar.radio("Mode d'entrée :", ("Texte", "Voix"))
    user_query = ""

    if option == "Texte":
        user_query = st.text_input("Posez votre question :", placeholder="Écrivez ici...")
    else:
        if st.button("🎤 Lancer la capture vocale"):
            with st.spinner("Écoute en cours..."):
                user_query = transcribe_speech()
                st.session_state.voice_input = user_query

        if "voice_input" in st.session_state:
            user_query = st.session_state.voice_input
            st.info(f"**Vous avez dit :** {user_query}")

    # Réponse
    if user_query and not user_query.startswith("Erreur"):
        with st.chat_message("assistant"):
            response = chatbot_response(user_query, sentences, vectorizer, tfidf_matrix)
            st.markdown(response)
else:
    st.info("👋 Bienvenue ! Veuillez charger un fichier texte pour commencer l'analyse.")
