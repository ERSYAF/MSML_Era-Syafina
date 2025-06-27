print(">>> TES: SKRIP automate_Era_Syafina.py MULAI DI SINI !!! <<<", flush=True)
import pandas as pd
import numpy as np
import ast
import os
import traceback # Impor modul traceback

def load_datasets(movies_path, credits_path):
    """Memuat dataset movies dan credits."""
    print(f"LOAD_DATASETS: Mencoba memuat movies dari: {movies_path}", flush=True)
    print(f"LOAD_DATASETS: Mencoba memuat credits dari: {credits_path}", flush=True)
    try:
        movies_df = pd.read_csv(movies_path)
        print("LOAD_DATASETS: Berhasil memuat movies_df.", flush=True)
        credits_df = pd.read_csv(credits_path)
        print("LOAD_DATASETS: Berhasil memuat credits_df.", flush=True)
        print("LOAD_DATASETS: Dataset movies dan credits berhasil dimuat (print asli).", flush=True)
        return movies_df, credits_df
    except FileNotFoundError as e_fnf:
        print(f"LOAD_DATASETS: FileNotFoundError saat memuat datasets: {e_fnf}", flush=True)
        return None, None
    except pd.errors.EmptyDataError as e_ede:
        print(f"LOAD_DATASETS: EmptyDataError (file CSV mungkin kosong atau format salah): {e_ede}", flush=True)
        return None, None
    except Exception as e_general:
        print(f"LOAD_DATASETS: TERJADI ERROR UMUM saat memuat datasets: {e_general}", flush=True)
        print(f"LOAD_DATASETS: Tipe error: {type(e_general)}", flush=True)
        traceback.print_exc()
        return None, None

def merge_datasets(movies_df, credits_df):
    """Menggabungkan dataset movies dan credits."""
    print("MERGE_DATASETS: Memulai penggabungan.", flush=True)
    if movies_df is None or credits_df is None:
        print("MERGE_DATASETS: movies_df atau credits_df adalah None, merge dibatalkan.", flush=True)
        return None

    if 'title' in credits_df.columns and 'title' in movies_df.columns:
        print("MERGE_DATASETS: Menghapus kolom 'title' dari credits_df.", flush=True)
        credits_df = credits_df.drop('title', axis=1, errors='ignore')
    merged_df = pd.merge(movies_df, credits_df, left_on='id', right_on='movie_id', how='inner')
    print("MERGE_DATASETS: Dataset berhasil digabungkan.", flush=True)
    return merged_df

def parse_json_like_column(column_str):
    """Helper function untuk mem-parse string yang mirip JSON."""
    try:
        return ast.literal_eval(column_str)
    except (ValueError, SyntaxError, TypeError):
        return []

def preprocess_json_columns(df, json_columns):
    """Memproses kolom-kolom berformat JSON."""
    print("PREPROCESS_JSON: Memproses kolom JSON...", flush=True)
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: parse_json_like_column(x) if isinstance(x, str) else [])
    print("PREPROCESS_JSON: Selesai memproses kolom JSON.", flush=True)
    return df

def extract_features(df):
    """Mengekstrak fitur baru seperti sutradara atau genre utama."""
    print("EXTRACT_FEATURES: Mengekstrak fitur...", flush=True)
    if 'crew' in df.columns:
        def get_director(crew_list):
            if not isinstance(crew_list, list): return None
            for member in crew_list:
                if isinstance(member, dict) and member.get('job') == 'Director':
                    return member.get('name')
            return None
        df['director'] = df['crew'].apply(get_director)
        print("EXTRACT_FEATURES: Kolom 'director' diekstrak.", flush=True)

    if 'genres' in df.columns:
        df['genres_processed'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) and x else [])
        print("EXTRACT_FEATURES: Kolom 'genres_processed' diekstrak.", flush=True)
    print("EXTRACT_FEATURES: Selesai mengekstrak fitur.", flush=True)
    return df

def handle_missing_values(df):
    """Menangani nilai yang hilang."""
    print("HANDLE_MISSING: Menangani missing values...", flush=True)
    if 'runtime' in df.columns:
        df['runtime'].fillna(df['runtime'].mean(), inplace=True)
    for col in ['homepage', 'tagline', 'overview']:
        if col in df.columns:
            df[col].fillna('', inplace=True)
    if 'director' in df.columns:
        df['director'].fillna('Unknown', inplace=True)
    print("HANDLE_MISSING: Selesai menangani missing values.", flush=True)
    return df

def convert_data_types(df):
    """Mengkonversi tipe data jika perlu."""
    print("CONVERT_DTYPES: Mengkonversi tipe data...", flush=True)
    if 'release_date' in df.columns and not df['release_date'].empty: # Tambah cek jika series kosong setelah dropna
        # Pastikan tidak ada NaN sebelum konversi, atau tangani NaT setelahnya
        df.dropna(subset=['release_date'], inplace=True)
        if not df['release_date'].empty: # Cek lagi setelah dropna
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            # Hapus baris yang menjadi NaT (Not a Time) setelah konversi yang gagal
            df.dropna(subset=['release_date'], inplace=True)
            if not df['release_date'].empty: # Cek sekali lagi
                df['release_year'] = df['release_date'].dt.year
                df['release_month'] = df['release_date'].dt.month
    print("CONVERT_DTYPES: Selesai mengkonversi tipe data.", flush=True)
    return df

def drop_duplicates_custom(df):
    """Menghapus data duplikat."""
    print("DROP_DUPLICATES: Menghapus duplikat...", flush=True)
    if 'id' in df.columns:
        df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    print("DROP_DUPLICATES: Selesai menghapus duplikat.", flush=True)
    return df

def run_preprocessing(raw_movies_path, raw_credits_path, output_path):
    """
    Fungsi utama untuk menjalankan semua langkah preprocessing.
    """
    print(f"RUN_PREPROCESSING: Fungsi dimulai. Menerima path:", flush=True)
    print(f"  movies_path = '{raw_movies_path}'", flush=True)
    print(f"  credits_path = '{raw_credits_path}'", flush=True)
    print(f"  output_path = '{output_path}'", flush=True)

    movies_df, credits_df = load_datasets(raw_movies_path, raw_credits_path)
    
    if movies_df is None or credits_df is None:
        print("RUN_PREPROCESSING: Gagal memuat dataset mentah dari load_datasets. Proses dihentikan.", flush=True)
        return None

    df = merge_datasets(movies_df, credits_df)
    if df is None:
        print("RUN_PREPROCESSING: Gagal menggabungkan dataset dari merge_datasets. Proses dihentikan.", flush=True)
        return None

    json_cols_to_process = ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages', 'cast', 'crew']
    df = preprocess_json_columns(df, json_cols_to_process)

    df = extract_features(df)

    df = handle_missing_values(df)

    df = convert_data_types(df)

    df = drop_duplicates_custom(df)

    columns_to_save = [
        'id', 'title', 'budget', 'revenue', 'runtime', 'popularity',
        'vote_average', 'vote_count', 'release_year', 'release_month',
        'overview', 'tagline', 'homepage', 'director', 'genres_processed'
    ]

    final_columns_to_save = [col for col in columns_to_save if col in df.columns]
    df_processed = df[final_columns_to_save].copy() 

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"RUN_PREPROCESSING: Membuat direktori output: {output_dir}", flush=True)
            os.makedirs(output_dir)
        df_processed.to_csv(output_path, index=False)
        print(f"RUN_PREPROCESSING: Dataset yang sudah diproses disimpan ke: {output_path}", flush=True)
        return df_processed
    except Exception as e:
        print(f"RUN_PREPROCESSING: Error saat menyimpan file: {e}", flush=True)
        traceback.print_exc()
        return None



if __name__ == '__main__':
    print(">>> TES: SKRIP automate_Era_Syafina.py MULAI DI SINI !!! <<<", flush=True)

    path_movies_raw = 'tmdb_5000_movies.csv'
    path_credits_raw = 'tmdb_5000_credits.csv'
    path_output_processed = 'Eksperimen_SML_Era-Syafina/Preprocessing/tmdb_movies_automated_processed.csv'


    print("Memulai proses preprocessing otomatis...", flush=True)
    print(f"  CWD saat ini (seharusnya): {os.getcwd()}", flush=True)
    print(f"  Mencoba memuat movies dari: {os.path.abspath(path_movies_raw)}", flush=True)
    print(f"  Mencoba memuat credits dari: {os.path.abspath(path_credits_raw)}", flush=True)
    print(f"  Akan menyimpan output ke: {os.path.abspath(path_output_processed)}", flush=True) # Debug path output
        
    processed_data = run_preprocessing(path_movies_raw, path_credits_raw, path_output_processed)

    if processed_data is not None:
        print("\nMAIN: Preprocessing otomatis selesai.", flush=True)
        print("MAIN: Beberapa baris data yang telah diproses:", flush=True)
        print(processed_data.head())
    else:
        print("\nMAIN: Preprocessing otomatis gagal.", flush=True)