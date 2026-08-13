# VeriDex: Proposed Pipeline Architecture

This document details the end-to-end architecture of the VeriDex fact-checking pipeline. The system employs a "hybrid" approach, combining static linguistic analysis with dynamic, live-web retrieval to provide robust and accurate verdicts.

## 1. Input Phase
The process begins when a user submits a claim (and optionally an associated image) via the VeriDex web interface. 
*   **Image Tampering Detection:** If an image is provided, it is immediately processed by a Vision Transformer (ViT) to determine if it exhibits artifacts of being AI-generated or tampered. 
*   **Text Processing:** The text statement is isolated and prepared for the core Natural Language Processing (NLP) pipeline.

## 2. Phase I: Linguistic Analysis (Fake News Model)
Before checking the factual truth of a claim, the system analyzes *how* the claim is written.
*   **Model:** A fine-tuned RoBERTa sequence classification model.
*   **Function:** The model scores the statement based purely on its linguistic structure, looking for sensationalism, hyperbole, clickbait phrasing, and other semantic markers commonly associated with deceptive news.
*   **Output:** A probability score indicating if the text is "Linguistically Suspicious" (Fake) or "Linguistically Sound" (Real).

## 3. Phase II: Live Context Retrieval (Tavily Search Engine)
Since the Fake News model only evaluates syntax and semantics, the system must retrieve external facts to verify the actual claim.
*   **Query Generation:** The claim is appended with the phrase "fact check" and dispatched as a search query via the Tavily Search API.
*   **Retrieval:** The API retrieves up to 3 highly relevant, live web articles, specifically excluding domains known for unfiltered user content (e.g., Facebook, Twitter, Reddit) to ensure higher-quality evidence.

## 4. Phase III: Evidence Corroboration (Stance Detection Model)
The system must now understand the relationship between the original claim and the retrieved articles.
*   **Model:** A fine-tuned DeBERTa-v3 Stance classification model.
*   **Function:** The original claim and the retrieved article snippets are passed as sentence pairs into the model. The model calculates whether the article supports (`PRO`) or refutes (`CON`) the claim.
*   **Anti-Dilution Logic (Keyword Debunking):** To prevent ambiguous news articles from diluting a strong debunk, the system scans snippets for explicit debunking keywords (e.g., "hoax", "false", "misinformation"). If found, the system aggressively boosts the `CON` confidence score.

## 5. Phase IV: Final Resolution Matrix
In the final step, VeriDex synthesizes the linguistic score (Phase I) and the stance evidence (Phases II & III) into a comprehensive verdict.

The resolution logic is as follows:

| Linguistic Analysis | Web Evidence Stance | Final Verdict | Explanation |
| :--- | :--- | :--- | :--- |
| **Sound** (Real) | **PRO** (Supported) | **Verified True** | The text is well-written and corroborated by live evidence. |
| **Suspicious** (Fake)| **CON** (Debunked) | **Verified Fake** | The text is deceptively written and explicitly debunked online. |
| **Suspicious** (Fake)| **PRO** (Supported) | **Mixed / Biased Truth** | The core facts are supported online, but the text is written using deceptive/clickbait tactics. |
| **Sound** (Real) | **CON** (Debunked) | **Polite Misinformation** | The text is written professionally and convincingly, but contradicts live factual evidence. |

If no articles are retrieved (e.g., a highly obscure claim), the system defaults to an **Unverified** state, flagging the claim as either suspicious or sound based solely on the Phase I linguistic score.

## Summary
By layering live-web stance detection on top of base linguistic analysis, VeriDex overcomes the static nature of standard machine learning models. It successfully intercepts "Polite Misinformation" (lies told cleanly) and identifies "Biased Truth" (facts told deceptively), resulting in a highly accurate, zero-shot verification engine.
