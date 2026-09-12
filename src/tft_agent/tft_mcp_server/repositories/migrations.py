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
    """
    CREATE TABLE datatft_snapshots (
        snapshot_id TEXT PRIMARY KEY, source TEXT NOT NULL, server TEXT NOT NULL,
        game_patch TEXT NOT NULL, sample_scope TEXT, source_url TEXT NOT NULL,
        retrieved_at TEXT NOT NULL, content_hash TEXT NOT NULL UNIQUE, etag TEXT,
        status TEXT NOT NULL, raw_payload TEXT NOT NULL
    );
    CREATE TABLE datatft_comps (
        snapshot_id TEXT NOT NULL, comp_id TEXT NOT NULL, name TEXT NOT NULL,
        tier TEXT, average_place REAL, play_rate REAL, top4_rate REAL, win_rate REAL,
        core_units TEXT NOT NULL, recommended_items TEXT NOT NULL, augments TEXT NOT NULL,
        level_plan TEXT NOT NULL, PRIMARY KEY(snapshot_id, comp_id),
        FOREIGN KEY(snapshot_id) REFERENCES datatft_snapshots(snapshot_id) ON DELETE CASCADE
    );
    CREATE INDEX idx_datatft_snapshot_patch ON datatft_snapshots(game_patch, retrieved_at);
    CREATE INDEX idx_datatft_comp_name ON datatft_comps(name COLLATE NOCASE);
    """,
    """
    CREATE TABLE static_entities (
        data_version TEXT NOT NULL, entity_id TEXT NOT NULL, entity_type TEXT NOT NULL,
        name TEXT NOT NULL, description TEXT NOT NULL, traits TEXT NOT NULL,
        components TEXT NOT NULL, tags TEXT NOT NULL, source TEXT NOT NULL,
        source_url TEXT NOT NULL, retrieved_at TEXT NOT NULL,
        PRIMARY KEY(data_version, entity_id)
    );
    CREATE INDEX idx_static_entities_name
        ON static_entities(name COLLATE NOCASE, data_version);
    CREATE INDEX idx_static_entities_type
        ON static_entities(entity_type, data_version);
    CREATE TABLE session_game_states (
        session_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, field_sources_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE session_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_session_messages_session
        ON session_messages(session_id, message_id);
    """,
    """
    CREATE TABLE canonical_entities (
        canonical_id TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        display_name TEXT NOT NULL,
        game_patch TEXT NOT NULL,
        set_number INTEGER NOT NULL,
        set_name TEXT NOT NULL,
        source TEXT NOT NULL,
        PRIMARY KEY(canonical_id, game_patch, set_number)
    );
    CREATE TABLE entity_aliases (
        alias TEXT NOT NULL COLLATE NOCASE,
        canonical_id TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        game_patch TEXT NOT NULL,
        set_number INTEGER NOT NULL,
        alias_kind TEXT NOT NULL,
        PRIMARY KEY(alias, canonical_id, game_patch, set_number)
    );
    CREATE INDEX idx_canonical_entities_scope_name
        ON canonical_entities(game_patch, set_number, display_name COLLATE NOCASE);
    CREATE INDEX idx_entity_aliases_scope_alias
        ON entity_aliases(game_patch, set_number, alias COLLATE NOCASE);
    """,
    """
    PRAGMA foreign_keys = OFF;
    BEGIN;

    ALTER TABLE datatft_comps RENAME TO datatft_comps_legacy;
    ALTER TABLE datatft_snapshots RENAME TO datatft_snapshots_legacy;
    DROP INDEX idx_datatft_snapshot_patch;
    DROP INDEX idx_datatft_comp_name;

    CREATE TABLE datatft_snapshots (
        snapshot_id TEXT PRIMARY KEY, source TEXT NOT NULL, server TEXT NOT NULL,
        game_patch TEXT NOT NULL, sample_scope TEXT, source_url TEXT NOT NULL,
        retrieved_at TEXT NOT NULL, content_hash TEXT NOT NULL, etag TEXT,
        status TEXT NOT NULL, raw_payload TEXT NOT NULL,
        sync_id TEXT,
        canonical_format_version INTEGER NOT NULL DEFAULT 1,
        raw_source_status TEXT NOT NULL DEFAULT 'LEGACY_UNAVAILABLE',
        validation_report TEXT,
        validated_at TEXT,
        UNIQUE(game_patch, content_hash)
    );
    CREATE INDEX idx_datatft_snapshot_patch
        ON datatft_snapshots(game_patch, retrieved_at);

    CREATE TABLE datatft_comps (
        snapshot_id TEXT NOT NULL, comp_id TEXT NOT NULL, name TEXT NOT NULL,
        tier TEXT, average_place REAL, play_rate REAL, top4_rate REAL, win_rate REAL,
        core_units TEXT NOT NULL, recommended_items TEXT NOT NULL, augments TEXT NOT NULL,
        level_plan TEXT NOT NULL, source_response_id TEXT,
        PRIMARY KEY(snapshot_id, comp_id),
        FOREIGN KEY(snapshot_id) REFERENCES datatft_snapshots(snapshot_id) ON DELETE CASCADE
    );
    CREATE INDEX idx_datatft_comp_name ON datatft_comps(name COLLATE NOCASE);

    INSERT INTO datatft_snapshots (
        snapshot_id, source, server, game_patch, sample_scope, source_url,
        retrieved_at, content_hash, etag, status, raw_payload
    )
    SELECT snapshot_id, source, server, game_patch, sample_scope, source_url,
           retrieved_at, content_hash, etag, status, raw_payload
    FROM datatft_snapshots_legacy;

    INSERT INTO datatft_comps (
        snapshot_id, comp_id, name, tier, average_place, play_rate, top4_rate,
        win_rate, core_units, recommended_items, augments, level_plan
    )
    SELECT snapshot_id, comp_id, name, tier, average_place, play_rate, top4_rate,
           win_rate, core_units, recommended_items, augments, level_plan
    FROM datatft_comps_legacy;

    DROP TABLE datatft_comps_legacy;
    DROP TABLE datatft_snapshots_legacy;

    CREATE TABLE game_patches (
        game_patch TEXT PRIMARY KEY,
        patch_family TEXT NOT NULL,
        set_number INTEGER NOT NULL CHECK(set_number >= 1),
        set_name TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('candidate','active','historical')),
        active_snapshot_id TEXT,
        source TEXT NOT NULL,
        discovered_at TEXT NOT NULL,
        synced_at TEXT,
        validated_at TEXT,
        activated_at TEXT,
        FOREIGN KEY(active_snapshot_id) REFERENCES datatft_snapshots(snapshot_id)
    );
    CREATE UNIQUE INDEX idx_game_patches_single_active
        ON game_patches(status) WHERE status = 'active';

    CREATE TABLE datatft_sync_runs (
        sync_id TEXT PRIMARY KEY,
        requested_patch TEXT,
        detected_patch TEXT,
        set_number INTEGER,
        set_name TEXT,
        server TEXT NOT NULL,
        status TEXT NOT NULL CHECK(
            status IN ('running','fetched','validated','activated','failed')
        ),
        started_at TEXT NOT NULL,
        completed_at TEXT,
        canonical_snapshot_id TEXT,
        warning_json TEXT NOT NULL DEFAULT '[]',
        failure_stage TEXT,
        failure_code TEXT,
        failure_message TEXT,
        FOREIGN KEY(canonical_snapshot_id) REFERENCES datatft_snapshots(snapshot_id)
    );

    CREATE TABLE datatft_source_responses (
        response_id TEXT PRIMARY KEY,
        sync_id TEXT NOT NULL,
        game_patch TEXT,
        endpoint_type TEXT NOT NULL,
        request_url TEXT NOT NULL,
        request_key TEXT,
        retrieved_at TEXT NOT NULL,
        http_status INTEGER,
        etag TEXT,
        content_hash TEXT NOT NULL,
        content_type TEXT,
        response_body TEXT NOT NULL,
        parse_status TEXT NOT NULL,
        error_message TEXT,
        FOREIGN KEY(sync_id) REFERENCES datatft_sync_runs(sync_id) ON DELETE CASCADE
    );
    CREATE INDEX idx_datatft_responses_sync
        ON datatft_source_responses(sync_id, endpoint_type, request_key);

    COMMIT;
    PRAGMA foreign_keys = ON;
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
