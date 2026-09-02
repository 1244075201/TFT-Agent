"""Ordered SQLite schema migrations kept out of tools and query methods."""

import sqlite3

MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS champion_syncs (
        data_version TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        champion_count INTEGER NOT NULL CHECK (champion_count >= 0)
    );
    CREATE TABLE IF NOT EXISTS champions (
        data_version TEXT NOT NULL,
        champion_id TEXT NOT NULL,
        name TEXT NOT NULL,
        cost INTEGER NOT NULL CHECK (cost >= 1),
        image_full TEXT NOT NULL,
        image_sprite TEXT NOT NULL,
        image_group TEXT NOT NULL,
        image_x INTEGER NOT NULL,
        image_y INTEGER NOT NULL,
        image_width INTEGER NOT NULL,
        image_height INTEGER NOT NULL,
        source TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        PRIMARY KEY (data_version, champion_id),
        FOREIGN KEY (data_version) REFERENCES champion_syncs(data_version) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_champions_name_version
        ON champions(name COLLATE NOCASE, data_version);
    """,
    """
    CREATE TABLE shop_odds (
        game_patch TEXT NOT NULL,
        source_version TEXT NOT NULL,
        level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 10),
        cost_1_probability REAL NOT NULL CHECK (cost_1_probability BETWEEN 0 AND 1),
        cost_2_probability REAL NOT NULL CHECK (cost_2_probability BETWEEN 0 AND 1),
        cost_3_probability REAL NOT NULL CHECK (cost_3_probability BETWEEN 0 AND 1),
        cost_4_probability REAL NOT NULL CHECK (cost_4_probability BETWEEN 0 AND 1),
        cost_5_probability REAL NOT NULL CHECK (cost_5_probability BETWEEN 0 AND 1),
        source TEXT NOT NULL,
        source_type TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        PRIMARY KEY (game_patch, source_version, level)
    );
    CREATE INDEX idx_shop_odds_patch_level ON shop_odds(game_patch, level);
    """,
    """
    CREATE TABLE knowledge_documents (
        document_id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
        source TEXT NOT NULL, source_type TEXT NOT NULL, document_type TEXT NOT NULL,
        game_patch TEXT, set_name TEXT, language TEXT NOT NULL, published_at TEXT,
        retrieved_at TEXT NOT NULL, url TEXT NOT NULL
    );
    CREATE TABLE knowledge_chunks (
        chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
        heading TEXT, text TEXT NOT NULL, embedding BLOB NOT NULL, embedding_model TEXT NOT NULL,
        embedding_model_revision TEXT, embedding_dimension INTEGER NOT NULL,
        embedded_at TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
        UNIQUE(document_id, chunk_index, embedding_model)
    );
    CREATE INDEX idx_knowledge_documents_patch ON knowledge_documents(game_patch);
    CREATE INDEX idx_knowledge_documents_set ON knowledge_documents(set_name);
    """,
    """
    ALTER TABLE knowledge_documents ADD COLUMN updated_at TEXT;
    CREATE INDEX idx_knowledge_documents_source_type
        ON knowledge_documents(source_type);
    CREATE INDEX idx_knowledge_documents_document_type
        ON knowledge_documents(document_type);
    """,
    """
    ALTER TABLE knowledge_documents ADD COLUMN strategy_comp_names TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE knowledge_documents ADD COLUMN strategy_entities TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE knowledge_documents ADD COLUMN strategy_topics TEXT NOT NULL DEFAULT '[]';
    """,
]


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply each migration exactly once inside a transaction."""

    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
    for version, script in enumerate(MIGRATIONS, start=1):
        if version in applied:
            continue
        connection.executescript(script)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) "
            "VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            (version,),
        )
    connection.commit()
