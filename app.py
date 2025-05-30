
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar datos
libros = pd.read_excel("dataset_libros_completo100 libros.xlsx", engine="openpyxl")
libros["Subject"] = libros["Programa/Carrera"] + " - " + libros["Materias"]

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(libros["Subject"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(libros.index, index=libros["Título"]).drop_duplicates()

def recomendar_libros(titulo, n=5):
    idx = indices[titulo]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    libro_indices = [i[0] for i in sim_scores]
    return libros[["Título", "Facultad", "Programa/Carrera", "Materias"]].iloc[libro_indices]

# Interfaz
st.title("📚 Recomendador de Libros UNAB")
titulo = st.selectbox("Selecciona un libro:", libros["Título"].unique())
num = st.slider("¿Cuántas recomendaciones deseas?", 1, 10, 5)

if st.button("Recomendar"):
    recomendaciones = recomendar_libros(titulo, n=num)
    st.write("Libros recomendados:")
    st.dataframe(recomendaciones)
