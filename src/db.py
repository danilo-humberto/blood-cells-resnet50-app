import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

DB_PATH = "data/interactions.db"


@st.cache_resource
def get_connection():
    """
    Cria (se não existir) e devolve uma conexão com o SQLite.
    Também garante a criação da tabela de interações.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model_name TEXT,
            predicted_class TEXT,
            confidence REAL,
            prob_eosinophil REAL,
            prob_lymphocyte REAL,
            prob_monocyte REAL,
            prob_neutrophil REAL,
            filename TEXT
        )
        """
    )
    conn.commit()
    return conn


def log_interaction(
    model_name: str,
    predicted_class: str,
    confidence: float,
    prob_eosinophil: float,
    prob_lymphocyte: float,
    prob_monocyte: float,
    prob_neutrophil: float,
    filename: str | None = None,
):
    """
    Insere um registro de interação na tabela.
    """
    conn = get_connection()
    ts = datetime.now().isoformat(timespec="seconds")

    conn.execute(
        """
        INSERT INTO interactions (
            created_at,
            model_name,
            predicted_class,
            confidence,
            prob_eosinophil,
            prob_lymphocyte,
            prob_monocyte,
            prob_neutrophil,
            filename
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            model_name,
            predicted_class,
            float(confidence),
            float(prob_eosinophil),
            float(prob_lymphocyte),
            float(prob_monocyte),
            float(prob_neutrophil),
            filename,
        ),
    )
    conn.commit()


def get_last_interactions(limit: int = 20) -> pd.DataFrame:
    """
    Retorna um DataFrame com as últimas interações registradas.
    """
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT
            id,
            created_at,
            model_name,
            predicted_class,
            ROUND(confidence * 100, 2) AS confidence_percent,
            prob_eosinophil,
            prob_lymphocyte,
            prob_monocyte,
            prob_neutrophil,
            filename
        FROM interactions
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(limit,),
    )
    return df