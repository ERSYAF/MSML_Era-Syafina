print(">>> TES: SKRIP Modelling.py MULAI DI SINI !!! <<<", flush=True)

import pandas as pd
import numpy as np
import ast
import mlflow
import mlflow.sklearn
import argparse
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text_for_token(text):
    if isinstance(text, str):
        return text.replace(" ", "").lower()
    return ""

def create_content_soup(df):
    print("CREATE_CONTENT_SOUP: Memulai pembuatan 'content soup'...", flush=True)
    df_copy = df.copy()

    required_cols = ['overview', 'genres_processed', 'director']
    for col in required_cols:
        if col not in df_copy.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset.")

    df_copy['overview_processed'] = df_copy['overview'].fillna('').astype(str).str.lower()

    def process_genres_list(genre_list):
        if isinstance(genre_list, list):
            return " ".join([clean_text_for_token(genre) for genre in genre_list])
        elif isinstance(genre_list, str):
            try:
                actual_list = ast.literal_eval(genre_list)
                return " ".join([clean_text_for_token(genre) for genre in actual_list])
            except:
                return ""
        return ""

    df_copy['genres_soup'] = df_copy['genres_processed'].apply(process_genres_list)
    df_copy['director_soup'] = df_copy['director'].apply(lambda x: clean_text_for_token(x) if pd.notnull(x) else "")

    df_copy['soup'] = (
        df_copy['overview_processed'] + ' ' +
        df_copy['genres_soup'] + ' ' +
        df_copy['director_soup']
    )

    print("CREATE_CONTENT_SOUP: 'Content soup' berhasil dibuat.", flush=True)
    return df_copy[['id', 'title', 'soup']]

def load_data_and_generate_soup(dataset_path):
    print(f"LOAD_AND_SOUP: Memuat dataset dari: {dataset_path}", flush=True)
    try:
        df = pd.read_csv(dataset_path)
        print(f"LOAD_AND_SOUP: Dataset berhasil dimuat. Baris: {len(df)}, Kolom: {len(df.columns)}", flush=True)
        return create_content_soup(df)
    except Exception as e:
        print(f"LOAD_AND_SOUP: ERROR - {e}", flush=True)
        return None

def get_recommendations(movie_title, cosine_sim_matrix_input, data, movie_indices, top_n=10):
    if movie_title not in movie_indices:
        return pd.Series(dtype='object')
    idx = movie_indices[movie_title]
    sim_scores = list(enumerate(cosine_sim_matrix_input[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices_output = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices_output]


if __name__ == "__main__":
    print(">>> BLOK __main__ di Modelling.py TERPANGGIL <<<", flush=True)

    parser = argparse.ArgumentParser(description="Content-based Movie Recommender Training with MLflow")
    parser.add_argument(
        '--dataset_path',
        type=str,
        default="Membangun_Model/tmdb_movies_automated_processed.csv",
        help='Path ke file dataset preprocessing hasil otomatisasi'
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    mlflow.set_experiment("Movie Recommender - Content Based")
    mlflow.sklearn.autolog()

    print(f"MAIN: Mulai proses dengan dataset: {dataset_path}", flush=True)
    movie_data_with_soup = load_data_and_generate_soup(dataset_path)

    if movie_data_with_soup is not None and not movie_data_with_soup.empty:
        with mlflow.start_run(run_name="ContentBasedRecommender_Run1"):
            tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=3, max_df=0.7)
            tfidf_matrix = tfidf.fit_transform(movie_data_with_soup['soup'].fillna(''))

            cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

            joblib.dump(tfidf, "tfidf_vectorizer.pkl")
            np.savez_compressed("cosine_matrix.npz", cosine_matrix=cosine_sim_matrix)

            mlflow.log_artifact("tfidf_vectorizer.pkl")
            mlflow.log_artifact("cosine_matrix.npz")

            X_dummy, y_dummy = make_classification(n_samples=200, n_features=20, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42)

            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            mlflow.sklearn.log_model(clf, "dummy_classifier")
            mlflow.log_metric("accuracy_dummy", acc)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix (Dummy Model)")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")

            indices = pd.Series(movie_data_with_soup.index, index=movie_data_with_soup['title']).drop_duplicates()
            test_movie_title = movie_data_with_soup['title'].iloc[0]
            recommendations = get_recommendations(test_movie_title, cosine_sim_matrix, movie_data_with_soup, indices)

            print(f"\n--- Rekomendasi untuk '{test_movie_title}' ---")
            if not recommendations.empty:
                print("\n".join([f"{i+1}. {title}" for i, title in enumerate(recommendations)]))
            else:
                print("Tidak ada rekomendasi ditemukan.")
    else:
        print("ERROR: Data tidak tersedia atau kosong!", flush=True)

    print("\nSkrip modelling.py selesai.", flush=True)
