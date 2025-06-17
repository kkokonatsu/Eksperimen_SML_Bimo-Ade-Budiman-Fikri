import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import load, dump
import os
import mlflow
import mlflow.sklearn

RAW_DATA_FILENAME = 'namadataset_raw.csv' 
PREPROCESSOR_FILENAME = f'preprocessor.joblib' 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR) 

RAW_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, RAW_DATA_FILENAME) 
PREPROCESSOR_PATH = os.path.join(BASE_DIR, PREPROCESSOR_FILENAME) 

PREPROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'namadataset_preprocessing')
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

print(f"Direktori Proyek: {PROJECT_ROOT_DIR}")
print(f"Direktori Preprocessing: {BASE_DIR}")
print(f"Path Data Mentah: {RAW_DATA_PATH}")
print(f"Path Preprocessor: {PREPROCESSOR_PATH}")
print(f"Direktori Output Data Preprocessing: {PREPROCESSED_DATA_DIR}")

mlflow.set_experiment("Preprocessing_Pipeline") 

with mlflow.start_run(run_name=f"Preprocessing_Run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
    mlflow.log_param("raw_data_filename", RAW_DATA_FILENAME)
    mlflow.log_param("preprocessor_filename", PREPROCESSOR_FILENAME)
    mlflow.log_param("test_size_split", 0.2)
    mlflow.log_param("random_state_split", 42)
    
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"\nBerhasil memuat data mentah dari: {RAW_DATA_PATH}")
        print(f"Bentuk data: {df.shape}")
        print("Kolom data mentah:", df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: File data mentah tidak ditemukan di '{RAW_DATA_PATH}'. Pastikan nama file dan path sudah benar.")
        exit() 
    except Exception as e:
        print(f"Error saat memuat data mentah: {e}")
        exit()

    if 'Personality' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le_personality = LabelEncoder()
        le_personality.fit(['Extrovert', 'Introvert']) 
        df['Personality_encoded'] = le_personality.transform(df['Personality'])
        print(f"Variabel target 'Personality' berhasil di-encode.")
        print(f"Mapping: {list(le_personality.classes_)} -> {list(range(len(le_personality.classes_)))}")
    else:
        print("Kolom 'Personality' (target) tidak ditemukan. Pastikan nama kolom target sudah benar.")
        exit()

    X = df.drop(['Personality', 'Personality_encoded'], axis=1) 
    y = df['Personality_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nData berhasil dibagi menjadi training ({X_train.shape[0]} baris) dan testing ({X_test.shape[0]} baris).")

    try:
        loaded_preprocessor = load(PREPROCESSOR_PATH)
        print(f"\nPreprocessor pipeline berhasil dimuat dari '{PREPROCESSOR_PATH}'.")
    except FileNotFoundError:
        print(f"Error: File preprocessor tidak ditemukan di '{PREPROCESSOR_PATH}'. Harap pastikan Anda telah melatih dan menyimpan preprocessor sebelumnya.")
        exit()
    except Exception as e:
        print(f"Error saat memuat preprocessor: {e}")
        exit()

    print("\nMelakukan preprocessing pada data pelatihan (X_train)...")
    X_train_processed = loaded_preprocessor.transform(X_train)
    print("Preprocessing X_train selesai.")

    print("Melakukan preprocessing pada data pengujian (X_test)...")
    X_test_processed = loaded_preprocessor.transform(X_test)
    print("Preprocessing X_test selesai.")

    print(f"Ukuran X_train setelah preprocessing: {X_train_processed.shape}")
    print(f"Ukuran X_test setelah preprocessing: {X_test_processed.shape}")

    X_train_processed_df = pd.DataFrame(X_train_processed)
    X_test_processed_df = pd.DataFrame(X_test_processed)

    print("\nMenyimpan data hasil preprocessing...")

    train_X_output_filename = os.path.join(PREPROCESSED_DATA_DIR, 'namadataset_preprocessing_X_train.parquet')
    test_X_output_filename = os.path.join(PREPROCESSED_DATA_DIR, 'namadataset_preprocessing_X_test.parquet')
    train_y_output_filename = os.path.join(PREPROCESSED_DATA_DIR, 'namadataset_preprocessing_y_train.csv')
    test_y_output_filename = os.path.join(PREPROCESSED_DATA_DIR, 'namadataset_preprocessing_y_test.csv')


    try:
        X_train_processed_df.to_csv(train_X_output_filename, index=False)
        print(f"- X_train_processed disimpan di: {train_X_output_filename}")

        X_test_processed_df.to_csv(test_X_output_filename, index=False)
        print(f"- X_test_processed disimpan di: {test_X_output_filename}")

        y_train.to_csv(train_y_output_filename, index=False, header=True)
        print(f"- y_train disimpan di: {train_y_output_filename}")

        y_test.to_csv(test_y_output_filename, index=False, header=True)
        print(f"- y_test disimpan di: {test_y_output_filename}")

    except Exception as e:
        print(f"Error saat menyimpan data hasil preprocessing: {e}")

print(f"\n--- Preprocessing Selesai ---")