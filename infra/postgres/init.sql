-- ReelMind PostgreSQL Schema
-- Stores entity metadata, features, and experiment results

-- ─── User & Content Entities ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    user_id         VARCHAR(20) PRIMARY KEY,
    age_group       VARCHAR(10) NOT NULL,
    country         CHAR(2) NOT NULL,
    region          VARCHAR(10),
    signup_ts       BIGINT NOT NULL,
    engagement_level FLOAT DEFAULT 0.5,
    is_creator      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_users_country ON users(country);
CREATE INDEX idx_users_engagement ON users(engagement_level);

CREATE TABLE IF NOT EXISTS creators (
    creator_id      VARCHAR(20) PRIMARY KEY,
    user_id         VARCHAR(20) REFERENCES users(user_id),
    follower_count  INTEGER DEFAULT 0,
    total_videos    INTEGER DEFAULT 0,
    avg_quality     FLOAT DEFAULT 0.5,
    creator_tier    VARCHAR(10) DEFAULT 'micro',
    engagement_rate FLOAT DEFAULT 0.05,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS videos (
    video_id        VARCHAR(20) PRIMARY KEY,
    creator_id      VARCHAR(20) REFERENCES creators(creator_id),
    duration_ms     INTEGER NOT NULL,
    category        VARCHAR(20) NOT NULL,
    upload_ts       BIGINT NOT NULL,
    quality_score   FLOAT DEFAULT 0.5,
    virality_score  FLOAT DEFAULT 0.0,
    total_views     INTEGER DEFAULT 0,
    total_likes     INTEGER DEFAULT 0,
    total_shares    INTEGER DEFAULT 0,
    avg_watch_pct   FLOAT DEFAULT 0.0,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_videos_category ON videos(category);
CREATE INDEX idx_videos_creator ON videos(creator_id);
CREATE INDEX idx_videos_upload ON videos(upload_ts DESC);
CREATE INDEX idx_videos_virality ON videos(virality_score DESC);

-- ─── Feature Store Tables ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS user_features (
    user_id             VARCHAR(20) PRIMARY KEY REFERENCES users(user_id),
    watch_count_1d      INTEGER DEFAULT 0,
    watch_count_7d      INTEGER DEFAULT 0,
    watch_count_30d     INTEGER DEFAULT 0,
    avg_watch_pct_7d    FLOAT DEFAULT 0.0,
    like_rate_7d        FLOAT DEFAULT 0.0,
    share_rate_7d       FLOAT DEFAULT 0.0,
    skip_rate_7d        FLOAT DEFAULT 0.0,
    avg_session_len     FLOAT DEFAULT 0.0,
    top_category_1      VARCHAR(20),
    top_category_2      VARCHAR(20),
    top_category_3      VARCHAR(20),
    last_active_ts      BIGINT,
    updated_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_features (
    video_id            VARCHAR(20) PRIMARY KEY REFERENCES videos(video_id),
    ctr_1d              FLOAT DEFAULT 0.0,
    ctr_7d              FLOAT DEFAULT 0.0,
    completion_rate_1d  FLOAT DEFAULT 0.0,
    completion_rate_7d  FLOAT DEFAULT 0.0,
    share_rate_7d       FLOAT DEFAULT 0.0,
    freshness_hours     FLOAT DEFAULT 0.0,
    momentum_score      FLOAT DEFAULT 0.0,
    updated_at          TIMESTAMP DEFAULT NOW()
);

-- ─── Experiment Tracking ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL UNIQUE,
    description     TEXT,
    status          VARCHAR(20) DEFAULT 'draft',
    model_version   VARCHAR(50),
    policy_version  VARCHAR(50),
    config_json     JSONB,
    started_at      TIMESTAMP,
    ended_at        TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_results (
    result_id       SERIAL PRIMARY KEY,
    experiment_id   INTEGER REFERENCES experiments(experiment_id),
    metric_name     VARCHAR(50) NOT NULL,
    metric_value    FLOAT NOT NULL,
    variant         VARCHAR(20) DEFAULT 'control',
    sample_size     INTEGER,
    confidence      FLOAT,
    recorded_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_exp_results ON experiment_results(experiment_id, metric_name);

-- ─── Model Registry (supplement to MLflow) ────────────────────────────────

CREATE TABLE IF NOT EXISTS model_deployments (
    deployment_id   SERIAL PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,
    model_version   VARCHAR(50) NOT NULL,
    stage           VARCHAR(20) DEFAULT 'staging',
    endpoint        VARCHAR(200),
    config_json     JSONB,
    deployed_at     TIMESTAMP DEFAULT NOW(),
    retired_at      TIMESTAMP
);

-- ─── Drift Detection Logs ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS drift_logs (
    log_id          SERIAL PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,
    drift_type      VARCHAR(30) NOT NULL,
    severity        VARCHAR(10) NOT NULL,
    metric_name     VARCHAR(50),
    baseline_value  FLOAT,
    current_value   FLOAT,
    psi_score       FLOAT,
    action_taken    VARCHAR(100),
    detected_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_drift_model ON drift_logs(model_name, detected_at DESC);
