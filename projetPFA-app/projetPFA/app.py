import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import plotly.express as px
import os

# Configuration de la page
st.set_page_config(
    page_title="Senti - Analyse de Sentiment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger les assets
def load_assets():
    try:
        logo = Image.open('assets/logo.png')
        return logo
    except FileNotFoundError:
        st.error("Logo non trouvé dans assets/logo.png")
        return None

# Charger le modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/classifier.pkl')
        vectorizer = joblib.load('model/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Modèle non trouvé. Veuillez d'abord générer les fichiers classifier.pkl et vectorizer.pkl")
        return None, None

# Style CSS personnalisé
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Fichier CSS {file_name} non trouvé")

# Fonction d'analyse de sentiment
def analyze_sentiment(text, model, vectorizer):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return prediction, proba

# Affichage des résultats
def display_results(prediction, proba):
    st.markdown("---")
    st.subheader("Résultat de l'analyse")
    
    sentiment_score = proba[1] - proba[0]
    
    if prediction == "positive":
        st.success(f"Sentiment: Positif 😊 (Confiance: {max(proba)*100:.2f}%)")
        gauge_color = "green"
    elif prediction == "negative":
        st.error(f"Sentiment: Négatif 😞 (Confiance: {max(proba)*100:.2f}%)")
        gauge_color = "red"
    else:
        st.warning(f"Sentiment: Neutre 😐 (Confiance: {max(proba)*100:.2f}%)")
        gauge_color = "orange"
    
    fig = px.bar(
        x=[sentiment_score],
        y=["Sentiment"],
        orientation='h',
        range_x=[-1, 1],
        color_discrete_sequence=[gauge_color],
        width=700,
        height=100
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Détails des probabilités"):
        proba_df = pd.DataFrame({
            "Sentiment": ["Négatif", "Neutre", "Positif"],
            "Probabilité": proba
        })
        st.dataframe(proba_df.style.format({"Probabilité": "{:.2%}"}))
        
        fig_pie = px.pie(
            proba_df,
            names="Sentiment",
            values="Probabilité",
            title="Répartition des probabilités",
            color="Sentiment",
            color_discrete_map={
                "Négatif": "#FF5252",
                "Neutre": "#FFD166",
                "Positif": "#06D6A0"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Application principale
def main():
    # Charger les assets et le modèle
    logo = load_assets()
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.stop()
    
    local_css("assets/style.css")
    
    # En-tête personnalisé
    st.markdown("<style>div.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    with col1:
        if logo:
            st.image(logo, width=100)
    with col2:
        st.title("Senti")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        if logo:
            st.image(logo, width=200)
        st.title("Senti")
        st.markdown("""
        **Senti** est une application d'analyse de sentiment qui utilise l'IA pour déterminer 
        si un texte exprime un sentiment positif, négatif ou neutre.
        """)
        
        st.markdown("---")
        st.markdown("### Comment ça marche?")
        st.markdown("""
        1. Entrez votre texte dans la zone de saisie
        2. Cliquez sur 'Analyser'
        3. Obtenez le résultat et les statistiques
        """)
        
        st.markdown("---")
        st.markdown("### Fonctionnalités")
        st.markdown("""
        - Analyse en temps réel
        - Historique des analyses
        - Visualisation des résultats
        - Précision élevée
        """)
        
        st.markdown("---")
        st.markdown("Développé avec ❤️ par [Votre Équipe]")
    
    # Onglets
    tab1, tab2, tab3 = st.tabs(["Analyse", "Historique", "À propos"])
    
    with tab1:
        st.header("Analyse de Texte")
        user_input = st.text_area("Entrez votre texte ici:", height=150, 
                                placeholder="Ex: J'adore ce produit! Il est incroyable...")
        
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            analyze_btn = st.button("Analyser le sentiment", type="primary")
        with col2:
            clear_btn = st.button("Effacer")
        
        if clear_btn:
            st.experimental_rerun()
        
        if analyze_btn and user_input:
            with st.spinner("Analyse en cours..."):
                prediction, proba = analyze_sentiment(user_input, model, vectorizer)
                display_results(prediction, proba)
                
                # Sauvegarder dans l'historique
                if "history" not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    "text": user_input,
                    "sentiment": prediction,
                    "confidence": max(proba),
                    "proba": proba,
                    "timestamp": pd.Timestamp.now()
                })
    
    with tab2:
        st.header("Historique des Analyses")
        if "history" in st.session_state and st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
            
            # Filtres
            col1, col2 = st.columns(2)
            with col1:
                sentiment_filter = st.multiselect(
                    "Filtrer par sentiment",
                    options=["positive", "negative", "neutral"],
                    default=["positive", "negative", "neutral"]
                )
            with col2:
                date_filter = st.date_input(
                    "Filtrer par date",
                    value=(
                        history_df["timestamp"].min().date(),
                        history_df["timestamp"].max().date()
                    )
                )
            
            # Appliquer les filtres
            filtered_df = history_df[
                (history_df["sentiment"].isin(sentiment_filter)) &
                (history_df["timestamp"].dt.date >= date_filter[0]) &
                (history_df["timestamp"].dt.date <= date_filter[1])
            ]
            
            # Afficher l'historique filtré
            if not filtered_df.empty:
                st.dataframe(
                    filtered_df.sort_values("timestamp", ascending=False),
                    column_config={
                        "text": "Texte",
                        "sentiment": "Sentiment",
                        "confidence": st.column_config.NumberColumn(
                            "Confiance",
                            format="%.2f%%"
                        ),
                        "timestamp": "Date/Heure"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Statistiques
                st.subheader("Statistiques")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total d'analyses", len(filtered_df))
                
                with col2:
                    positive_count = len(filtered_df[filtered_df["sentiment"] == "positive"])
                    st.metric("Positifs", f"{positive_count} ({positive_count/len(filtered_df):.1%})")
                
                with col3:
                    negative_count = len(filtered_df[filtered_df["sentiment"] == "negative"])
                    st.metric("Négatifs", f"{negative_count} ({negative_count/len(filtered_df):.1%})")
                
                # Graphique d'évolution
                time_df = filtered_df.set_index("timestamp").resample("D")["sentiment"].value_counts().unstack().fillna(0)
                fig = px.line(
                    time_df,
                    x=time_df.index,
                    y=time_df.columns,
                    title="Évolution des sentiments au fil du temps",
                    labels={"value": "Nombre d'analyses", "timestamp": "Date"},
                    color_discrete_map={
                        "positive": "#06D6A0",
                        "negative": "#FF5252",
                        "neutral": "#FFD166"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune analyse ne correspond aux filtres sélectionnés.")
        else:
            st.info("Aucune analyse dans l'historique pour le moment.")
    
    with tab3:
        st.header("À propos de Senti")
        st.markdown("""
        ### Notre Mission
        Senti a été créé pour aider les entreprises et les particuliers à comprendre les sentiments exprimés 
        dans les textes, que ce soit des avis clients, des commentaires sur les réseaux sociaux ou tout autre 
        contenu textuel.
        
        ### Technologie
        Notre application utilise des algorithmes avancés de machine learning et de traitement du langage naturel 
        pour fournir des analyses précises en temps réel.
        
        - **Modèle**: Classificateur entraîné sur des milliers d'exemples
        - **Précision**: Plus de 85% sur nos jeux de test
        - **Langues supportées**: Principalement le français et l'anglais
        
        ### Cas d'utilisation
        - Analyse des avis clients
        - Surveillance des réseaux sociaux
        - Études de marché
        - Recherche académique
        
        ### Contact
        Pour toute question ou suggestion, contactez-nous à contact@senti.ai
        """)
        
        st.markdown("---")
        st.subheader("Notre Équipe")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image("https://via.placeholder.com/150", width=150)
            st.markdown("**Jean Dupont**\n\nDirecteur Technique")
        
        with col2:
            st.image("https://via.placeholder.com/150", width=150)
            st.markdown("**Marie Martin**\n\nData Scientist")
        
        with col3:
            st.image("https://via.placeholder.com/150", width=150)
            st.markdown("**Pierre Lambert**\n\nDéveloppeur Full-Stack")

if __name__ == "__main__":
    main()