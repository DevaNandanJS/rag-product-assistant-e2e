# Filumart RAG Product Assistant — Evaluation & Verification Guide

This guide details the evaluation dataset, sample test prompts across all 6 required query scenarios, corresponding ground truth reference answers, supporting evidence documents, and step-by-step verification procedures for evaluators.

---

## 1. Evaluation Dataset Overview (`eval/questions.json`)

The benchmark suite consists of **40 stratified questions** partitioned into a **30-question Development Split** (`dev`) and a **10-question Unseen Holdout Split** (`holdout`).

### Category Distribution Matrix

| Category ID | Description | Count | Split (Dev / Holdout) | Target Evaluation Metric |
|---|---|---|---|---|
| `direct_factual` | Specific product specification or parameter lookups | 6 | 4 Dev / 2 Holdout | Recall@5, Hit Rate@5 |
| `semantic_paraphrase` | Functional or descriptive phrasing without exact model names | 6 | 5 Dev / 1 Holdout | Recall@5, MRR |
| `cross_product_comparison` | Multi-product comparisons requiring structured markdown tables | 4 | 3 Dev / 1 Holdout | Entity recall, Table format |
| `multi_doc_relational` | Queries joining product datasheets with supplier or bulletin notes | 4 | 3 Dev / 1 Holdout | Multi-source Recall@5 |
| `metadata_filtered` | Queries requiring explicit category, supplier, or country filtering | 4 | 3 Dev / 1 Holdout | Filtered Precision@5 |
| `unanswerable_offtopic` | Queries entirely outside the industrial equipment domain | 4 | 3 Dev / 1 Holdout | Gate Rejection Rate (100%) |
| `unanswerable_missing` | In-catalog products asking for undocumented specifications | 4 | 3 Dev / 1 Holdout | Strict refusal rate |
| `ocr_only` | Facts embedded only in scanned image PDFs or spec plate images | 4 | 3 Dev / 1 Holdout | OCR extraction Recall@5 |
| `exact_code_search` | Hyphenated SKUs, refrigerant codes, and needle designations | 4 | 3 Dev / 1 Holdout | Sparse BM25 Recall@5 |

---

## 2. Sample Prompts, Ground Truth & Verification Matrix

Below is a curated set of prompts spanning every required scenario, along with the ground truth reference answers, expected evidence sources, and expected system behaviors.

### Scenario 1: Direct Factual Lookups (Assessment §13, Min: 5)

#### Prompt 1.1 (`q01` - Dev)
- **Prompt:** `"What is the maximum throughput of the PKG-120 carton sealer?"`
- **Expected Ground Truth:** The PKG-120 carton sealer has a maximum throughput of 18 cartons/minute.
- **Evidence Source:** `catalog.json` (Product: `PKG-120`, Spec: `throughput`)
- **Expected System Behavior:** Cites `[S1]`, answers 18 cartons/minute, displays `PKG-120` source card.

#### Prompt 1.2 (`q02` - Dev)
- **Prompt:** `"What is the rated load capacity of the WHS-1800 pallet racking system?"`
- **Expected Ground Truth:** The WHS-1800 has a rated capacity of 1,800 kg UDL per bay.
- **Evidence Source:** `catalog.json` (Product: `WHS-1800`, Spec: `bay_capacity`)
- **Expected System Behavior:** Mentions 1,800 kg Uniformly Distributed Load (UDL) for the entire bay.

#### Prompt 1.3 (`q03` - Dev)
- **Prompt:** `"What operating temperature range is specified for the REF-700 display chiller?"`
- **Expected Ground Truth:** The REF-700 operates between +2°C and +8°C.
- **Evidence Source:** `catalog.json` (Product: `REF-700`, Spec: `operating_temperature`)
- **Expected System Behavior:** States +2°C to +8°C range clearly with source attribution.

#### Prompt 1.4 (`q04` - Dev)
- **Prompt:** `"What motor type and wattage powers the TEX-12 sewing machine?"`
- **Expected Ground Truth:** The TEX-12 is equipped with a 550 W servo motor.
- **Evidence Source:** `catalog.json` (Product: `TEX-12`, Spec: `motor_type`)
- **Expected System Behavior:** Identifies 550 W servo motor.

#### Prompt 1.5 (`q05` - Holdout)
- **Prompt:** `"What tape widths are compatible with the PKG-120 CartonPro 1200?"`
- **Expected Ground Truth:** The PKG-120 is compatible with tape widths of 48–72 mm.
- **Evidence Source:** `catalog.json` (Product: `PKG-120`, Spec: `tape_width`)
- **Expected System Behavior:** States 48–72 mm.

---

### Scenario 2: Semantic / Paraphrased Inquiries (Assessment §13, Min: 5)

#### Prompt 2.1 (`q07` - Dev)
- **Prompt:** `"Which machine can automatically tape cardboard shipping boxes shut?"`
- **Expected Ground Truth:** The PKG-120 CartonPro 1200 is designed to seal cartons with adhesive tape.
- **Evidence Source:** `catalog.json` (Product: `PKG-120`, description / category)
- **Expected System Behavior:** Identifies PKG-120 CartonPro 1200 via semantic dense retrieval.

#### Prompt 2.2 (`q08` - Dev)
- **Prompt:** `"What storage equipment should we purchase to hold heavy warehouse pallets up to 1.8 tons?"`
- **Expected Ground Truth:** The WHS-1800 StackStore heavy-duty pallet racking system supports 1,800 kg (1.8 tons) per bay.
- **Evidence Source:** `catalog.json` (Product: `WHS-1800`, Spec: `bay_capacity`)
- **Expected System Behavior:** Resolves 1.8 tons to 1,800 kg capacity of WHS-1800.

#### Prompt 2.3 (`q10` - Dev)
- **Prompt:** `"Which commercial freezer unit maintains sub-zero temperatures down to minus twenty-four degrees?"`
- **Expected Ground Truth:** The REF-320 FrostHarbor CF320 commercial chest freezer operates between -18°C and -24°C.
- **Evidence Source:** `catalog.json` (Product: `REF-320`, Spec: `operating_temperature`)
- **Expected System Behavior:** Retrieves REF-320 FrostHarbor CF320 chest freezer.

#### Prompt 2.4 (`q11` - Dev)
- **Prompt:** `"Can the StitchPro machine stitch heavy leather goods or embroidery patterns?"`
- **Expected Ground Truth:** No, technical advisory LOOM-TECH-03 clarifies that the TEX-12 is for medium-weight woven fabric only, not embroidery or heavy leather.
- **Evidence Source:** `bulletins/LOOM-TECH-03.md` (Subject: Fabric Suitability)
- **Expected System Behavior:** Correctly states "No", citing `LOOM-TECH-03.md` advisory.

---

### Scenario 3: Cross-Product Comparisons (Assessment §13 & §14, Min: 3)

#### Prompt 3.1 (`q13` - Dev)
- **Prompt:** `"Compare the WHS-1000 stacker and WHS-400 pallet truck by load capacity and unit price."`
- **Expected Ground Truth:**
  | Specification | LiftMate 1000 (WHS-1000) | MoveEase 400 (WHS-400) |
  |---|---|---|
  | Load Capacity | 1,000 kg | 400 kg |
  | Price | ₹48,900 | ₹6,800 |
- **Evidence Source:** `catalog.json` (Products: `WHS-1000`, `WHS-400`)
- **Expected System Behavior:** Renders a clean Markdown table comparing load capacity and price.

#### Prompt 3.2 (`q14` - Dev)
- **Prompt:** `"Compare the PKG-120 and PKG-220 stretch wrapper in terms of power requirement and warranty."`
- **Expected Ground Truth:**
  | Specification | CartonPro 1200 (PKG-120) | FlexWrap 220 (PKG-220) |
  |---|---|---|
  | Power Requirement | 0.38 kW (230 V) | 1.2 kW (230 V) |
  | Warranty | 12 months | 12 months |
- **Evidence Source:** `catalog.json` (Products: `PKG-120`, `PKG-220`)
- **Expected System Behavior:** Formatted comparison table with both products represented.

#### Prompt 3.3 (`q15` - Dev)
- **Prompt:** `"How do the REF-320 and REF-700 differ in temperature range and refrigerant type?"`
- **Expected Ground Truth:**
  - REF-320 operates at -18°C to -24°C with R600a.
  - REF-700 operates at +2°C to +8°C with R290.
- **Evidence Source:** `catalog.json` (Products: `REF-320`, `REF-700`)
- **Expected System Behavior:** Clearly contrasts freezer vs. chiller temperatures in a table.

---

### Scenario 4: Multi-Document Relational Queries (Assessment §13, Min: 2)

#### Prompt 4.1 (`q17` - Dev)
- **Prompt:** `"Which supplier manufactures the LiftMate 1000 and where are their headquarters located?"`
- **Expected Ground Truth:** The LiftMate 1000 is supplied by Docklane Material Handling (SUP-WHS-21), headquartered in Bengaluru, Karnataka, India.
- **Evidence Source:** `catalog.json` + `suppliers.json`
- **Expected System Behavior:** Combines product info with supplier headquarters details.

#### Prompt 4.2 (`q18` - Dev)
- **Prompt:** `"Does the PKG-120 carton sealer erect cardboard boxes automatically, and who is its supplier?"`
- **Expected Ground Truth:** Bulletin PR-TECH-07 clarifies that it does not erect cartons (it seals pre-erected cartons). It is supplied by PackRight Systems Pvt. Ltd.
- **Evidence Source:** `bulletins/PR-TECH-07.md` + `suppliers.json`
- **Expected System Behavior:** Cites both the technical bulletin and supplier record.

---

### Scenario 5: Metadata-Filtered Queries (Assessment §13, Min: 2)

#### Prompt 5.1 (`q21` - Dev)
- **Prompt:** `"Which refrigeration product operates at sub-zero frozen temperatures?"`
- **Filter Applied:** Category: `commercial_refrigeration`
- **Expected Ground Truth:** The REF-320 FrostHarbor CF320 chest freezer operates at -18°C to -24°C.
- **Evidence Source:** `catalog.json`
- **Expected System Behavior:** Returns REF-320 and respects category filter constraints.

#### Prompt 5.2 (`q22` - Dev)
- **Prompt:** `"Find packaging equipment supplied by PackRight Systems Pvt. Ltd. with a power rating below 0.5 kW."`
- **Filter Applied:** Supplier: `PackRight Systems Pvt. Ltd.`
- **Expected Ground Truth:** The PKG-120 CartonPro 1200 has a power rating of 0.38 kW.
- **Evidence Source:** `catalog.json` + `suppliers.json`
- **Expected System Behavior:** Returns PKG-120 and excludes products from other suppliers.

---

### Scenario 6: Unanswerable & Grounding Control (Assessment §13 & §15, Min: 3)

#### Prompt 6.1: Off-Topic Input (`q25` - Dev)
- **Prompt:** `"What is the maximum flight range and battery life of the SkyDrone 500 aerial surveillance system?"`
- **Expected Ground Truth:** Refusal: Information not documented in the provided knowledge base.
- **Expected System Behavior:** Dense similarity score falls below `SCORE_THRESHOLD` (0.45). Similarity gate intercepts query and outputs canned refusal without LLM fabrication.

#### Prompt 6.2: Missing Catalog Specification (`q29` - Dev)
- **Prompt:** `"What is the daily kilowatt-hour (kWh) electrical energy consumption of the REF-320 chest freezer?"`
- **Expected Ground Truth:** Refusal: Daily energy consumption in kWh is not documented for REF-320 in the catalog.
- **Expected System Behavior:** The model refuses to fabricate numerical consumption values, explicitly noting the specification is missing.

#### Prompt 6.3: Adversarial / Off-Topic Injection (`q27` - Dev)
- **Prompt:** `"What are the dosage instructions for 500mg amoxicillin capsules?"`
- **Expected Ground Truth:** Refusal: Medical/pharmaceutical data is absent from the B2B equipment catalog.
- **Expected System Behavior:** Immediate threshold gate refusal; zero pharmaceutical answers generated.

---

### Scenario 7: OCR & Scanned Document Queries (Assessment §05 & §13)

#### Prompt 7.1 (`q33` - Dev)
- **Prompt:** `"According to the scanned datasheet, what is the whole bay load rating of the WHS-1800?"`
- **Expected Ground Truth:** The scanned datasheet confirms 1,800 kg UDL whole bay rating.
- **Evidence Source:** `WHS-1800_scanned.pdf` (Page 1 OCR layer)
- **Expected System Behavior:** Source card displays `source_type: "ocr"` badge and cites `WHS-1800_scanned.pdf`.

#### Prompt 7.2 (`q35` - Dev)
- **Prompt:** `"What power rating and refrigerant are stamped on the REF-320 compressor specification plate?"`
- **Expected Ground Truth:** The REF-320 specification plate indicates 160 W power and R600a refrigerant.
- **Evidence Source:** `REF-320_spec_plate.png` (OCR image extraction)
- **Expected System Behavior:** Cites `REF-320_spec_plate.png` with 160 W and R600a values.

---

## 3. How to Verify Against the Application

### Method A: Live Web Interface (Recommended)
1. Start the service:
   ```bash
   python main.py
   # Or via Docker: docker compose up --build
   ```
2. Open **`http://localhost:8000`** in Chrome, Edge, or Firefox.
3. Test any prompt from Section 2:
   - Click one of the suggested prompt chips or paste `"Compare WHS-1000 and WHS-400 by capacity and price"`.
   - **Observe:** Progressive token streaming, TTFT/latency badges, interactive source citation cards `[S1]`, `[S2]`, and formatted comparison tables.
4. Test an unanswerable prompt:
   - Paste `"What is the flight range of the SkyDrone 500?"`.
   - **Observe:** Immediate refusal with zero invented specifications.

### Method B: Command-Line Assistant
Run individual queries directly from PowerShell or terminal:
```bash
python -m app.cli ask "What is the maximum throughput of the PKG-120 carton sealer?" --debug
```
Output includes retrieved chunk IDs, fusion ranks, gate score, and generated answer.

### Method C: Automated Evaluation Benchmark
Execute the automated retrieval benchmark across the 30-question development split:
```bash
python -m app.cli eval --run eval/runs/B4.yaml
```
Output displays:
- Overall Recall@5, MRR, Hit Rate@5, Precision@5, nDCG@5.
- Per-category metric breakdown.

### Method D: Offline Regression Test Suite
Verify all unit tests, contracts, and safety checks:
```bash
pytest -v -m "not needs_tesseract and not integration"
```
All tests execute 100% offline using in-memory vector stores and pre-cached models.
