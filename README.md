# SemEval-2027 Task 1: RETECO (Track 2: Conversational Retrieval & Grounded Generation)

Repository modulare, riproducibile e ottimizzata per la partecipazione ufficiale a **SemEval-2027 Task 1: RETECO** (*Reasoning-Oriented Retrieval with Temporal & Conversational Context*), focalizzata su **Track 2** (Sub-track **2a: Conversational Retrieval** e Sub-track **2b: Grounded Generation with Gold Passages**).

---

## 1. Panoramica del Task: SemEval-2027 Task 1 (RETECO)

I benchmark tradizionali di Information Retrieval (es. BEIR) valutano i sistemi unicamente sulla similarità semantica superficiale tra query e documento. Nella realtà applicativa dei sistemi RAG (*Retrieval-Augmented Generation*) e degli assistenti virtuali, la rilevanza dipende da fattori contestuali complessi: **quando** un fatto è valido, **come** cambiano le circostanze nel tempo e **cosa è già stato stabilito** nei turni precedenti di una conversazione.

**RETECO** è la competizione condivisa accettata a SemEval-2027 per colmare questa lacuna, suddivisa in due percorsi indipendenti e complementari:

* **Track 1 · Temporal Grounded Retrieval (TEMPO):** richiede di recuperare passaggi vincolati da requisiti temporali espliciti o impliciti (Sub-track 1a per query globali e 1b per step decomposti).
* **Track 2 · Conversational Retrieval & RAG (RECOR):** richiede di tracciare l'evoluzione dello stato del dialogo, gestire coreferenze e anafore, e compiere inferenze logiche multi-step per identificare passaggi rilevanti su corpora verticali di dominio.

### Team Organizzatore Ufficiale

RETECO è coordinato da ricercatori della **University of Innsbruck**, della **University of British Columbia (UBC)** e della **Johns Hopkins University**:

* **Abdelrahman Abdallah** (University of Innsbruck — Lead Organizer)
* **Mohammed Ali** (University of Innsbruck — Co-organizer, First Author di RECOR)
* **Muhammad Abdul-Mageed** (University of British Columbia)
* **Kevin Duh** (Johns Hopkins University)
* **Adam Jatowt** (University of Innsbruck)

---

## 2. Approfondimento Track 2: RECOR Benchmark

Track 2 si basa sul benchmark scientifico **RECOR** (*Reasoning-focused Multi-turn Conversational Retrieval Benchmark*, Findings of ACL 2026). La traccia affronta il collasso prestazionale dei motori di ricerca convenzionali all'aumentare della profondità del dialogo ($T_1 \dots T_5+$).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CONVERSATION FLOW (RECOR)                       │
├────────────────────────────────────────────────────────────────────────┤
│ Turn 1: "In FPV drone motors, why is lubrication only for bearings?"   │
│         └── Context: No previous conversation.                         │
│                                                                        │
│ Turn 2: "What happens if oil gets on the stator coils?"                │
│         └── Context: Resolves implicit reference to Turn 1 motors.     │
│                                                                        │
│ Turn 3: "Does high KV exacerbate that failure?"                        │
│         └── Context: Coreference ("that failure") + Domain reasoning.  │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │      SUB-TRACK 2a       │             │      SUB-TRACK 2b       │
    │ Conversational Retr.    │             │ Grounded Generation     │
    ├─────────────────────────┤             ├─────────────────────────┤
    │ Search domain corpus    │             │ Gold passages provided  │
    │ Output: Top-10 Passages │             │ Output: Factual Answer  │
    │ Metric: nDCG@10         │             │ Metric: 5-Dim LLM-Judge │
    └─────────────────────────┘             └─────────────────────────┘

```

### Sub-track 2a: Conversational Retrieval

* **Obiettivo:** Dato l'intero storico del dialogo pregresso (`conversation_history`) e la domanda del turno target (`query`), il sistema deve interrogare l'intero corpus del dominio di riferimento ed estrarre una graduatoria ordinata dei migliori 10 passaggi rilevanti.
* **Sfide Tecniche:**
1. *Risoluzione del contesto discorsivo:* gestione di ellissi, continuazioni tematiche e pronomi/anafore orfane di referente esplicito.
2. *Ragionamento multi-step:* i passaggi gold supportano la risposta tramite implicazione logica, non attraverso il mero overlap lessicale di parole chiave.
3. *Decadimento per profondità di turno:* capacità di mantenere stabili le metriche anche su turni avanzati ($T_3, T_4, T_5+$).


* **Input del Sistema:** file `benchmark_train.json` o `benchmark_dev.json` (conversazioni e cronologia) + `documents.jsonl` (corpus integrale del dominio).
* **Output Richiesto:** file in formato TREC a 6 colonne contenente esattamente i primi 10 passaggi per ciascun identificatore di turno.
* **Metrica Ufficiale di Valutazione:** **$nDCG@10$** (Normalized Discounted Cumulative Gain al cutoff 10), calcolato mediante la libreria ufficiale `pytrec_eval` e macro-mediato su tutti i domini.

### Sub-track 2b: Grounded Generation with Gold Passages

* **Obiettivo:** Data la cronologia conversazionale, il turno corrente e l'insieme dei passaggi gold documentali *esplicitamente forniti a priori*, il modello generativo deve sintetizzare una risposta corretta, coerente e interamente ancorata alle evidenze fornite.
* **Scopo Sperimentale:** Isolare la qualità della componente linguistica generativa dagli errori di richiamo del retriever, stabilendo l'upper-bound teorico della pipeline.
* **Input del Sistema:** file `benchmark_*.json` contenente per ciascun turno il campo `gold_doc_ids`, abbinato ai testi dei passaggi estratti da `documents.jsonl`.
* **Output Richiesto:** stringa testuale della risposta per ciascun turno conversazionale.
* **Metrica Ufficiale di Valutazione:** Valutazione multidimensionale tramite LLM-as-a-Judge (GPT-4o) calibrato su scala Likert 1–5 normalizzata [0, 1] lungo 5 dimensioni: **Correctness**, **Completeness**, **Relevance**, **Coherence** e **Faithfulness**.

---

## 3. Architettura dei Dati e Corpus RECOR

L'intero dataset v1.1 di RETECO è ospitato su Hugging Face: [`DataScience-UIBK/RETECO-SemEval2027`](https://www.google.com/search?q=https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027&utm_source=gemini).

Track 2 comprende **507.141 documenti** distribuiti su **11 domini specialistici indipendenti**:

* **6 Domini accademico-scientifici (derivati da BRIGHT):** Biology, Earth Science, Economics, Psychology, Robotics, Sustainable Living.
* **5 Domini tecnici ad alta competenza (derivati da StackExchange):** Drones, Hardware, Law, Medical Sciences, Politics.

| Dominio | Documenti Corpus | Conversazioni (Tr/Dv) | Turni Target (Tr/Dv) | Qrels Gold (Tr/Dv) |
| --- | --- | --- | --- | --- |
| `biology` | 57,359 | 59 / 26 | 247 / 115 | 368 / 196 |
| `drones` | 16,381 | 26 / 11 | 104 / 38 | 227 / 107 |
| `earth_science` | 121,249 | 69 / 29 | 321 / 133 | 536 / 181 |
| `economics` | 50,220 | 52 / 22 | 196 / 92 | 497 / 159 |
| `hardware` | 26,308 | 32 / 14 | 130 / 58 | 281 / 114 |
| `law` | 20,027 | 35 / 15 | 164 / 66 | 441 / 145 |
| `medicalsciences` | 23,297 | 31 / 13 | 139 / 44 | 305 / 101 |
| `politics` | 16,712 | 30 / 13 | 159 / 54 | 385 / 142 |
| `psychology` | 52,835 | 59 / 25 | 234 / 99 | 529 / 191 |
| `robotics` | 61,961 | 48 / 20 | 184 / 75 | 310 / 147 |
| `sustainable_living` | 60,792 | 55 / 23 | 235 / 84 | 419 / 182 |
| **TOTALE TRACK 2** | **507,141** | **496 / 211** | **2,113 / 858** | **4,298 / 1,665** |

### Regole Fondamentali di Splitting

1. **Il corpus non è mai diviso:** in entrambi gli split (Train e Dev), il retriever esegue la ricerca sull'intero corpus del dominio (`documents.jsonl`).
2. **Split 70/30 a livello di intera conversazione:** tutti i turni appartenenti alla medesima conversazione risiedono nello stesso split. Questo schema previene qualsiasi *data leakage* o contaminazione contestuale tra train e dev.
3. **Gold judgments rilasciati per entrambi gli split pubblici:** i file `qrels_train.txt` e `qrels_dev.txt` sono noti e consentono di sviluppare, fittare e validare localmente il sistema.

---

## 4. Leaderboard, Piattaforma di Gara e Valutazione Ufficiale

### Dove si trova la Leaderboard e la Piattaforma di Sottomissione

* **Piattaforma di Competizione:** la fase di gara ufficiale sarà ospitata su **CodaLab / CodaBench**, con link e registrazione che verranno ufficializzati sul sito istituzionale [RETECO Participate](https://www.google.com/search?q=https://datascienceuibk.github.io/RETECO/participate.html&utm_source=gemini) e sulla [Mailing List Ufficiale](https://www.google.com/search?q=https://groups.google.com/g/semeval-2027-reteco&utm_source=gemini).
* **Finestra Temporale di Valutazione Ufficiale:** **10 – 31 Gennaio 2027**.
* **Test Set Nascosto (Cieco):** Durante la finestra di valutazione verrà rilasciato un nuovo set di test annotato e mai reso pubblico (~200 turni target per Track 2, bilanciati sugli 11 domini) privo di giudizi di rilevanza (`qrels`).
* **Politica di Sottomissione:** È previsto un tetto massimo di upload giornalieri per team (*daily submission cap*) per impedire il tuning sui dati di test.
* **Classifica Ufficiale:** La leaderboard stila la graduatoria dei sistemi partecipanti basandosi unicamente sul valore macro-mediato di **$nDCG@10$**. Le prestazioni disaggregate per dominio e per profondità di turno ($T_1 \dots T_5+$) sono riportate a fini diagnostici.

### Standard di Formattazione della Submission (Formato TREC a 6 Colonne)

Per la Sub-track 2a, la sottomissione deve essere un singolo file di testo `.trec` con esattamente 6 colonne separate da spazi o tabulazioni:

```text
<topic_id> Q0 <doc_id> <rank> <score> <tag>

```

* **`topic_id`:** identificatore obbligatorio nella forma `<conversation_id>_turn_<turn_id>` (es. `ex_3025_turn_1`).
* **`Q0`:** costante fissa dello standard TREC.
* **`doc_id`:** identificativo univoco del documento all'interno del dominio (es. `drones_ex_3025_doc_0`).
* **`rank`:** rango assegnato al documento, da `1` a `10`.
* **`score`:** punteggio numerico di pertinenza (deve essere strettamente non crescente all'aumentare del rango).
* **`tag`:** identificativo mnemonico del team o della run (es. `TEAM_RUN_1`).

La correttezza sintattica del file viene verificata dal tool ufficiale dello starter kit prima del caricamento sulla piattaforma:

```bash
python starter_kit/format_checker.py outputs/subtrack_2a/submission_2a.trec
# Output atteso: RESULT: VALID

```

---

## 5. Risorse Istituzionali e Riferimenti Ufficiali

* **Sito Web Ufficiale del Task:** [datascienceuibk.github.io/RETECO](https://www.google.com/search?q=https://datascienceuibk.github.io/RETECO/&utm_source=gemini)
* **Repository GitHub Ufficiale:** [DataScienceUIBK/RETECO](https://www.google.com/search?q=https://github.com/DataScienceUIBK/RETECO&utm_source=gemini)
* **Dataset Hugging Face (v1.1):** [DataScience-UIBK/RETECO-SemEval2027](https://www.google.com/search?q=https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027&utm_source=gemini)
* **Task Proposal Ufficiale (PDF):** [RETECO SemEval-2027 Proposal](https://www.google.com/search?q=https://datascienceuibk.github.io/RETECO/assets/papers/RETECO_SemEval_2027_Proposal.pdf&utm_source=gemini)
* **Starter Kit Ufficiale (Baselines & Scorer):** [DataScienceUIBK/RETECO/starter_kit](https://www.google.com/search?q=https://github.com/DataScienceUIBK/RETECO/tree/main/starter_kit&utm_source=gemini)
* **Mailing List dei Partecipanti:** [semeval-2027-reteco @ Google Groups](https://www.google.com/search?q=https://groups.google.com/g/semeval-2027-reteco&utm_source=gemini)
* **Paper Scientifico di Riferimento RECOR:** [ACL Anthology 2026.findings-acl.129](https://www.google.com/url?sa=E&source=gmail&q=https://aclanthology.org/2026.findings-acl.129/)

---

## 6. Baseline di Riferimento Ufficiali

Risultati ufficiali $nDCG@10$ (macro-average) pubblicati dagli organizzatori con modello BM25 standard ($k_1=0.9, b=0.4$):

| Sub-track | Configurazione Query | Split Train | Split Dev |
| --- | --- | --- | --- |
| **2a Conversational Retrieval** | *Current turn only* (senza cronologia) | 0.1837 | 0.1827 |
| **2a Conversational Retrieval** | *Turn + Conversation history* (dialogo completo) | **0.4539** | **0.4379** |

L'inclusione della cronologia conversazionale incrementa il punteggio di oltre **+139%**, attestando che il recupero contestualizzato è il fattore discriminante del task.

---

## 7. Citazioni Ufficiali

Se utilizzi questo codice o i benchmark di riferimento, cita i lavori ufficiali di TEMPO e RECOR:

```bibtex
@inproceedings{ali2026recor,
  title={{RECOR: Reasoning-focused Multi-turn Conversational Retrieval Benchmark}},
  author={Ali, Mohammed and Abdallah, Abdelrahman and Agarwal, Amit and Patel, Hitesh Laxmichand and Jatowt, Adam},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={2688--2723},
  year={2026},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2026.findings-acl.129},
  url={https://aclanthology.org/2026.findings-acl.129/}
}

@article{abdallah2026tempo,
  title={{TEMPO: A Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval}},
  author={Abdallah, Abdelrahman and Ali, Mohammed and Abdul-Mageed, Muhammad and Jatowt, Adam},
  journal={arXiv preprint arXiv:2601.09523},
  year={2026},
  url={https://arxiv.org/abs/2601.09523}
}
```