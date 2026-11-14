-- PostgreSQL Database Schema for DFI Prediction Platform

-- Clean up existing tables (for development/migrations)
DROP TABLE IF EXISTS feedback CASCADE;
DROP TABLE IF EXISTS prediction_history CASCADE;
DROP TABLE IF EXISTS model_metadata CASCADE;
DROP TABLE IF EXISTS dfi_labels CASCADE;
DROP TABLE IF EXISTS drug_compounds CASCADE;
DROP TABLE IF EXISTS food_compounds CASCADE;
DROP TABLE IF EXISTS foods CASCADE;
DROP TABLE IF EXISTS drugs CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 1. Users Table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Drugs Table
CREATE TABLE drugs (
    id SERIAL PRIMARY KEY,
    drugbank_id VARCHAR(50) UNIQUE,
    name VARCHAR(255) UNIQUE NOT NULL,
    smiles TEXT,
    formula VARCHAR(100),
    molecular_weight DOUBLE PRECISION,
    description TEXT,
    class_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Foods Table
CREATE TABLE foods (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    category VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Food Compounds Table
CREATE TABLE food_compounds (
    id SERIAL PRIMARY KEY,
    food_id INT REFERENCES foods(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    smiles TEXT,
    pubchem_cid VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Drug Compounds Table
CREATE TABLE drug_compounds (
    id SERIAL PRIMARY KEY,
    drug_id INT REFERENCES drugs(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    smiles TEXT,
    pubchem_cid VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Drug-Food Interaction Labels Table
CREATE TABLE dfi_labels (
    id SERIAL PRIMARY KEY,
    drug_id INT REFERENCES drugs(id) ON DELETE CASCADE,
    food_id INT REFERENCES foods(id) ON DELETE CASCADE,
    severity VARCHAR(50) NOT NULL CHECK (severity IN ('safe', 'moderate', 'severe')),
    mechanism TEXT,
    recommendation TEXT,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (drug_id, food_id)
);

-- 7. Prediction History Table
CREATE TABLE prediction_history (
    id SERIAL PRIMARY KEY,
    drug_name VARCHAR(255) NOT NULL,
    food_name VARCHAR(255) NOT NULL,
    smiles_drug TEXT,
    smiles_food TEXT,
    predicted_class INT NOT NULL, -- 0: Severe/Negative, 1: Safe/Positive, 2: Moderate/Non-sig
    confidence DOUBLE PRECISION NOT NULL,
    probabilities JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Model Metadata Table
CREATE TABLE model_metadata (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    accuracy DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    best_params JSONB,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT FALSE
);

-- 9. Feedback Table
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    prediction_id INT REFERENCES prediction_history(id) ON DELETE CASCADE,
    user_rating INT NOT NULL CHECK (user_rating BETWEEN 1 AND 5),
    user_comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for rapid lookup
CREATE INDEX idx_drugs_name ON drugs(name);
CREATE INDEX idx_foods_name ON foods(name);
CREATE INDEX idx_dfi_labels_pair ON dfi_labels(drug_id, food_id);
CREATE INDEX idx_pred_history_names ON prediction_history(drug_name, food_name);
