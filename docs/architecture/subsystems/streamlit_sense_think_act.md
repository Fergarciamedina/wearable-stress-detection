# Subsystem 8: Streamlit Sense–Think–Act Engine

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Architectural Concept: Closed-Loop Wearable Health

The Streamlit deployment dashboard ([`src/act/dashboard.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/act/dashboard.py)) implements an interactive closed-loop health prototype structured around the classic robotics **Sense–Think–Act** cycle:

1. **SENSE (Biometric Acquisition & Buffering):**
   - Continuously buffers incoming EDA and BVP streams in a time-indexed circular queue (`ReplaySensorAdapter`).
   - Extracts moving 30-second temporal windows with a configurable sliding stride (default 5.0 seconds).
   - Dynamically evaluates signal quality (e.g., checks sensor contact detachment or extreme values).
2. **THINK (Inference & State Estimation):**
   - Extracts the 20-dimensional physiological feature vector.
   - Loads the pre-trained deployment model (`deploy_RF_EDA+PPG.pkl`).
   - Outputs continuous stress probabilities and binary discrete states (`REST` vs. `STRESS`).
   - Surfaces empirical research metrics through a collapsible panel.
3. **ACT (Adaptive Contextual Intervention):**
   - Formulates a structured state dictionary:
     ```python
     context = {
         "model_prediction": 1,  # 0: REST, 1: STRESS
         "confidence": 0.82,
         "eda_mean_us": 4.15,
         "pulse_rate_bpm": 104.2,
         "motion_acc_magnitude": 1.05,
         "quality_status": "GOOD"
     }
     ```
   - Evaluates dual actuation engines:
     - **Deterministic Rules (`DeterministicRuleAct`):** $< 0.1\,\text{ms}$ latency, zero hallucinations, clinical wellness framing.
     - **LLM Natural Language Adapter (`LLMActAdapter`):** Generates empathetic personal advice with latency and cost tracking.

---

## 2. Subsystem Block Diagram

```mermaid
graph TD
    subgraph SENSE["1. SENSE: Ring Buffer Engine"]
        IN_STREAM["Sensor Input Stream<br/>(Replay / Synthetic / EmotiBit)"]
        RING_BUF["BaseSensorAdapter Circular Queue<br/>Time-indexed (ts, value) tuples"]
        PRUNE["Aggressive Buffer Pruning<br/>Drops samples older than t - 60s"]
        WIN_SLICE["30-Second Window Extraction<br/>eda_window (120), bvp_window (1920)"]
        IN_STREAM --> RING_BUF --> PRUNE --> WIN_SLICE
    end

    subgraph THINK["2. THINK: Real-Time Inference"]
        FEAT_ENG["FeatureExtractor<br/>-> 20D Feature Vector"]
        DEPLOY_RF["deploy_RF_EDA+PPG.pkl<br/>StandardScaler + RandomForest"]
        PROB_GAUGE["Posterior Class Probabilities<br/>P(STRESS), P(REST)"]
        WIN_SLICE --> FEAT_ENG --> DEPLOY_RF --> PROB_GAUGE
    end

    subgraph ACT["3. ACT: Dual Guidance Engines"]
        STATE_CTX["Structured State Object<br/>{pred, conf, eda_mean, bpm, acc, quality}"]
        
        subgraph RULES["Deterministic Rule Engine"]
            RULE_EVAL["DeterministicRuleAct<br/>Evaluates priority ladder<br/>Latency: < 0.1ms"]
            RULE_CARD["Action Code & Prescription<br/>(e.g., STRESS_BREATHING_BREAK)"]
            RULE_EVAL --> RULE_CARD
        end

        subgraph LLM["LLM Natural Language Adapter"]
            LLM_EVAL["LLMActAdapter (offline-mock-v1)<br/>Generates conversational advice"]
            LLM_CARD["Empathetic Text Advice Card<br/>Latency ~120ms | Cost: $0.00012"]
            LLM_EVAL --> LLM_CARD
        end

        STATE_CTX --> RULE_EVAL & LLM_EVAL
    end

    subgraph UI["Streamlit Presentation Dashboard"]
        COL1["Column 1: Waveforms (SENSE)"]
        COL2["Column 2: Probability Gauge (THINK)"]
        COL3["Column 3: Recommendations (ACT)"]
        LOGGER[("Session Log<br/>llm_logs/replay_session_log.json")]
    end

    PROB_GAUGE --> STATE_CTX
    WIN_SLICE --> COL1
    PROB_GAUGE --> COL2
    RULE_CARD & LLM_CARD --> COL3 --> LOGGER

    style SENSE fill:#e8f4f8,stroke:#0c5460
    style THINK fill:#d1ecf1,stroke:#0c5460
    style ACT fill:#fff3cd,stroke:#856404
    style UI fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 3. Subsystem Execution Flowchart

```mermaid
flowchart TD
    START(["Streamlit Event Loop Trigger"]):::terminator --> STEP_BTN{"User clicks 'Step Forward'?"}:::decision

    STEP_BTN -->|"No"| IDLE(["Wait for user action"]):::terminator
    STEP_BTN -->|"Yes"| ADV_BUFFER["adapter.advance_by_step(step_s)"]:::process

    ADV_BUFFER --> CHECK_LEN{"adapter.has_sufficient_data()?"}:::decision
    CHECK_LEN -->|"False"| WARN_BUF["Display: Buffer accumulating..."]:::process
    CHECK_LEN -->|"True"| EXTRACT_WIN["latest_win = adapter.get_latest_window()"]:::process

    EXTRACT_WIN --> RENDER_SENSE["Plot biometric lines: EDA (µS) and BVP (raw)"]:::output

    EXTRACT_WIN --> RUN_THINK["Compute 20D feature vector<br/>Load deploy_RF_EDA+PPG.pkl<br/>Run predict_proba()"]:::subproc

    RUN_THINK --> RENDER_THINK["Render Gauge: P(STRESS) % & Label State"]:::output

    RENDER_THINK --> ASSEMBLE_CTX["Assemble context dictionary:<br/>- model_prediction<br/>- confidence<br/>- eda_mean_us<br/>- pulse_rate_bpm<br/>- motion_acc_magnitude<br/>- quality_status"]:::process

    ASSEMBLE_CTX --> RUN_RULES["DeterministicRuleAct.generate_recommendation(ctx)"]:::subproc
    ASSEMBLE_CTX --> RUN_LLM["LLMActAdapter.generate_recommendation(ctx)"]:::subproc

    RUN_RULES & RUN_LLM --> RENDER_ACT["Display Rule Action & LLM Message cards"]:::output

    RENDER_ACT --> CHECK_LOG{"User clicked 'Save Anonymized Step Log'?"}:::decision
    CHECK_LOG -->|"Yes"| DUMP_LOG[("Write JSON log entry to llm_logs/")]:::storage
    CHECK_LOG -->|"No"| FINISH_LOOP(["Step Cycle Completed"]):::terminator
    DUMP_LOG --> FINISH_LOOP

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
    classDef storage fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px,color:#721c24;
    classDef output fill:#e2e3e5,stroke:#383d41,stroke-width:1.5px,color:#383d41;
```
