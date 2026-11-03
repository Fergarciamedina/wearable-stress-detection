# Subsystem 5: Similarity-Aware Contextualization (Hypothesis 2)

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Scientific Hypothesis & Formulation

### Hypothesis 2 (H2)
Can non-parametric physiological exemplar retrieval detect physical exercise confounding *without* requiring an accelerometer?

If a 20-dimensional autonomic feature vector (EDA + PPG) is ambiguous to a parametric Random Forest boundary, querying an exact $k$-Nearest-Neighbors index populated with historical exemplars of verified REST (0), STRESS (1), and EXERCISE (2) provides non-parametric voting evidence of the physiological neighborhood.

---

## 2. Mathematical Formulation & Algorithmic Design

Implemented in [`src/research/similarity_aware/knn_retrieval.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/research/similarity_aware/knn_retrieval.py) (`PhysiologicalKNN`) and [`src/research/similarity_aware/similarity_context.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/research/similarity_aware/similarity_context.py) (`SimilarityContextualizer`):

### 1. Exemplar Space Assembly (Training Time)
Given $N-1$ development subjects:
- Labeled Stress Session Vectors: $X_{\text{rs}} \in \mathbb{R}^{N_{\text{rs}} \times 20}$, $y_{\text{rs}} \in \{0, 1\}$.
- Labeled Exercise Session Vectors: $X_{\text{ex}} \in \mathbb{R}^{N_{\text{ex}} \times 20}$, $y_{\text{ex}} = 2$ (EXERCISE).
- Reference Index:
  $$X_{\text{all}} = \begin{bmatrix} X_{\text{rs}} \\ X_{\text{ex}} \end{bmatrix}, \quad y_{\text{all}} = \begin{bmatrix} y_{\text{rs}} \\ \mathbf{2}_{N_{\text{ex}}} \end{bmatrix}$$
- Indexing: Standardized via training scaler, stored in an exact spatial Ball-Tree:
  $$\text{Index} = \text{NearestNeighbors}(k=7, \text{metric}=\text{'euclidean'}, \text{algorithm}=\text{'ball\_tree'})$$

### 2. $k$-NN Voting Query
For candidate test feature vector $\mathbf{x}_{\text{test}} \in \mathbb{R}^{20}$:
1. Retrieve indices of $k=7$ nearest neighbors $\mathcal{N}_7(\mathbf{x}_{\text{test}})$.
2. Calculate empirical neighborhood fractions:
   $$f_{\text{ex}} = \frac{1}{7} \sum_{j \in \mathcal{N}_7} \mathbb{I}[y_j = 2], \quad f_{\text{stress}} = \frac{1}{7} \sum_{j \in \mathcal{N}_7} \mathbb{I}[y_j = 1], \quad f_{\text{rest}} = \frac{1}{7} \sum_{j \in \mathcal{N}_7} \mathbb{I}[y_j = 0]$$

### 3. Contextualization Override Rule
With default threshold $\theta_{\text{ex}} = 0.40$ ($\ge 3$ of 7 neighbors):
$$\hat{y}_{i, \text{H2}} = \begin{cases} 
0, & \text{if } \hat{y}_i = 1 \text{ and } f_{\text{ex}}(\mathbf{x}_i) \ge 0.40 \\
\hat{y}_i, & \text{otherwise}
\end{cases}$$

---

## 3. Empirical Trade-off & Autonomic Overlap

- **Exercise Alarm Suppression:** Drops Exercise FAR from 61.58% down to **2.79%** (E2).
- **The Over-Suppression Vulnerability:** Acute cognitive stress and vigorous exercise induce severe overlapping sympathetic responses (elevated heart rate, depressed RMSSD, elevated tonic conductance). Consequently, many true cognitive stress windows fall near exercise clusters in 20D space, causing Stress F1 to drop from 0.4660 to **0.1784**, and Macro-F1 to decline to **0.5226**.

---

## 4. Subsystem Block Diagram

```mermaid
graph LR
    subgraph REF_INDEX["Reference Exemplar Pool (Training Only)"]
        TR_RS["Training Stress Sessions<br/>REST (y=0) & STRESS (y=1)"]
        TR_EX["Training Exercise Sessions<br/>Assigned EXERCISE (y=2)"]
        CONCAT_MAT["Concatenate Matrices<br/>X_all: (N_rs + N_ex, 20)<br/>y_all: (N_rs + N_ex,)"]
        BALL_TREE["Exact Ball-Tree Index<br/>StandardScaler + NearestNeighbors(k=7)"]
        TR_RS & TR_EX --> CONCAT_MAT --> BALL_TREE
    end

    subgraph QUERY_STAGE["Inference Query Phase"]
        Y_PRED["Primary RF Prediction<br/>y_pred in {0, 1}"]
        X_TEST["Test Physiological Vector<br/>x_test in R^20"]
        VOTE_CALC["k-NN Neighbor Query (k=7):<br/>- rest_frac = sum(lbl==0)/7<br/>- stress_frac = sum(lbl==1)/7<br/>- exercise_frac = sum(lbl==2)/7"]
        GATE_DEC{"y_pred == 1<br/>AND exercise_frac >= 0.40?"}
        OVERRIDE["Override: STRESS -> REST (0)<br/>[Physiological Exercise Match]"]
        PASSTHROUGH["Pass-Through: y_out = y_pred"]

        X_TEST --> VOTE_CALC
        BALL_TREE -.->|Spatial Index| VOTE_CALC
        VOTE_CALC & Y_PRED --> GATE_DEC
        GATE_DEC -->|True| OVERRIDE
        GATE_DEC -->|False| PASSTHROUGH
    end

    subgraph OUTPUT["Final Contextualized Output"]
        Y_OUT["y_contextualized in {0, 1}"]
        OVERRIDE & PASSTHROUGH --> Y_OUT
    end

    style REF_INDEX fill:#f8f9fa,stroke:#6c757d
    style QUERY_STAGE fill:#e8f4f8,stroke:#0c5460
    style GATE_DEC fill:#fff3cd,stroke:#856404
    style OVERRIDE fill:#d4edda,stroke:#28a745
    style OUTPUT fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 5. Subsystem Execution Flowchart

```mermaid
flowchart TD
    START(["Start: SimilarityContextualizer.contextualize(y_pred, X_test)"]):::terminator --> QUERY_BATCH["Execute Batch Query:<br/>votes = knn.query_votes_batch(X_test)"]:::subproc

    QUERY_BATCH --> SCALE_QUERY["Standardize X_test using knn._scaler.transform()"]:::process
    SCALE_QUERY --> SEARCH_TREE["dists, indices = knn._nn.kneighbors(X_test_scaled)"]:::process

    SEARCH_TREE --> EXTRACT_VOTES["For each test window i:<br/>labels = knn._labels[indices[i]]<br/>exercise_frac = count(labels == 2) / 7.0<br/>stress_frac = count(labels == 1) / 7.0<br/>rest_frac = count(labels == 0) / 7.0"]:::process

    EXTRACT_VOTES --> LOOP_DEC{"For each window i in N"}:::decision
    LOOP_DEC -->|"Next i"| CHK_ALARM{"y_pred[i] == 1 (STRESS)?"}:::decision

    CHK_ALARM -->|"No (REST)"| LEAVE_REST["y_ctx[i] = 0"]:::process
    LEAVE_REST --> LOOP_DEC

    CHK_ALARM -->|"Yes (STRESS)"| CHK_EX_VOTE{"exercise_frac >= 0.40?<br/>(>= 3 of 7 neighbors exercise)"}:::decision
    CHK_EX_VOTE -->|"Yes"| SUPPRESS["Override: y_ctx[i] = 0<br/>(Exercise similarity detected)"]:::process
    CHK_EX_VOTE -->|"No"| RETAIN["Retain: y_ctx[i] = 1<br/>(Pure stress neighborhood)"]:::process

    SUPPRESS & RETAIN --> LOOP_DEC
    LOOP_DEC -->|"All Windows Processed"| END_SIM(["Return Contextualized Predictions y_ctx"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
```
