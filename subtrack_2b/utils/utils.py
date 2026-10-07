#!/usr/bin/env python3
"""
subtrack_2b/utils/utils.py
Funzioni di utilità per il Sub-track 2b di RETECO / SemEval-2027:
- Gestione seed e logging deterministico
- Rilevamento hardware (CUDA, MPS, CPU)
- Costruzione del prompt e template di raffinamento
- Troncamento strutturato del contesto
- Metriche generative (ROUGE-L, METEOR, BERTScore) con import opzionali sicuri
"""

import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Fissa il seed deterministico su Python, NumPy e PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def detect_device(preferred: str = "auto") -> Tuple[torch.device, str]:
    """
    Rileva il miglior dispositivo di calcolo disponibile.
    Supporta CUDA, MPS (Apple Silicon) e CPU.
    """
    pref = preferred.lower().strip()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if pref == "cpu":
        return torch.device("cpu"), "cpu"

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Crea la directory se non esiste e restituisce l'oggetto Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Carica un file di configurazione YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Salva una struttura dati in formato JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Carica un file JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File JSON non trovato: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Salva una lista di record in formato JSONL."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Carica un file JSONL restituendo una lista di dizionari."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File JSONL non trovato: {p}")
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


def setup_logging(log_dir: Union[str, Path], run_tag: str, log_level: str = "INFO") -> logging.Logger:
    """Inizializza un logger con output simultaneo su console e file con timestamp."""
    ensure_dir(log_dir)
    logger = logging.getLogger("subtrack_2b")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        Path(log_dir) / f"{run_tag}_{timestamp}.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def normalize_text(text: str) -> str:
    """Pulisce e normalizza gli spazi nel testo."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Stima o calcola con precisione il numero di token in un testo."""
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return len(text.split())


def truncate_history(
    history: str,
    max_tokens: int,
    tokenizer: Optional[Any] = None,
) -> Tuple[str, bool]:
    """
    Tronca la cronologia mantenendo prioritariamente i turni più recenti.

    Supporta formati comuni:
        Turn N:
        Q: / A:
        User: / Assistant:
        Human: / AI:
        Customer: / Agent:
        Bot:

    La funzione lavora prima a livello di unità conversazionali e poi,
    quando necessario, a livello token.

    Restituisce:
        (history_truncated, was_truncated)
    """

    if not history or not history.strip():
        return "No previous conversation.", False

    clean_history = history.strip()

    if clean_history.lower() == "no previous conversation.":
        return "No previous conversation.", False

    if max_tokens <= 0:
        return "", True

    current_tokens = count_tokens(
        clean_history,
        tokenizer,
    )

    if current_tokens <= max_tokens:
        return clean_history, False

    # ------------------------------------------------------------------
    # 1. Identifica unità conversazionali.
    #
    # Ogni nuova domanda/turno inizia con uno dei marker seguenti.
    # Questo permette di mantenere Q+A insieme.
    # ------------------------------------------------------------------
    turn_pattern = re.compile(
        r"(?=^\s*(?:"
        r"Turn\s+\d+\s*:|"
        r"Q\s*:|"
        r"User\s*:|"
        r"Human\s*:|"
        r"Customer\s*:"
        r")\s*)",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    units = [
        block.strip()
        for block in turn_pattern.split(clean_history)
        if block.strip()
    ]

    # Se non abbiamo una struttura riconoscibile, prova a usare
    # i blocchi separati da righe vuote.
    if len(units) <= 1:
        units = [
            block.strip()
            for block in re.split(
                r"\n\s*\n+",
                clean_history,
            )
            if block.strip()
        ]

    # ------------------------------------------------------------------
    # 2. Mantieni dal fondo i blocchi più recenti senza superare il budget.
    # ------------------------------------------------------------------
    selected_reversed: List[str] = []
    accumulated_tokens = 0

    for unit in reversed(units):
        unit_tokens = count_tokens(
            unit,
            tokenizer,
        )

        if accumulated_tokens + unit_tokens <= max_tokens:
            selected_reversed.append(unit)
            accumulated_tokens += unit_tokens
        else:
            break

    if selected_reversed:
        selected = list(
            reversed(selected_reversed)
        )

        truncated_history = "\n\n".join(
            selected
        )

        return truncated_history.strip(), True

    # ------------------------------------------------------------------
    # 3. Fallback: il singolo blocco più recente può essere già troppo lungo.
    #    In quel caso fai token-level truncation mantenendo la parte finale.
    # ------------------------------------------------------------------
    latest_unit = units[-1] if units else clean_history

    if tokenizer is not None:
        try:
            ids = tokenizer.encode(
                latest_unit,
                add_special_tokens=False,
            )

            ids = ids[-max_tokens:]

            truncated_history = tokenizer.decode(
                ids,
                skip_special_tokens=True,
            ).strip()

            return truncated_history, True

        except Exception as exc:
            logging.getLogger("subtrack_2b").warning(
                "Tokenizer truncation fallita nella history: %s",
                exc,
            )

    # Fallback semplice senza tokenizer.
    words = latest_unit.split()

    truncated_history = " ".join(
        words[-max_tokens:]
    )

    return truncated_history.strip(), True

def build_prompt(
    query: str,
    conversation_history: str,
    evidence_passages: List[str],
    domain: Optional[str] = None,
    subquestion_reasoning: Optional[str] = None,
    ablation: str = "D",
    include_domain: bool = True,
    use_subquestion_reasoning: bool = False,
) -> str:
    """
    Costruisce il prompt strutturato per il modello generativo.
    Ablations supportate:
      - 'A': query only
      - 'B': query + history
      - 'C': query + evidence
      - 'D': query + history + evidence (default)
      - 'E': query + history + evidence + auxiliary reasoning
    """
    sections: List[str] = []

    # 1. System Instruction
    system_text = (
        "You are an expert grounded conversational assistant.\n"
        "Your task is to answer the user's current question accurately and directly.\n"
        "Rely strictly on the provided gold evidence passages for factual claims.\n"
        "Use the conversation history to resolve coreferences, ellipses, and dialogue context.\n"
        "Do not invent facts not supported by the evidence.\n"
        "Do not output internal reasoning steps, chain-of-thought, or metatalk like 'According to passage 1'.\n"
        "Provide only the final, cohesive, conversationally natural answer."
    )
    sections.append(f"SYSTEM:\n{system_text}")

    # 2. Domain Context
    if include_domain and domain:
        clean_domain = domain.replace("_", " ").title()
        sections.append(f"DOMAIN:\n{clean_domain}")

    # 3. Conversation History
    include_hist = ablation in ["B", "D", "E"]
    clean_hist = conversation_history.strip() if conversation_history else "No previous conversation."
    if include_hist:
        sections.append(f"CONVERSATION HISTORY:\n{clean_hist}")

    # 4. Gold Evidence
    include_evid = ablation in ["C", "D", "E"]
    if include_evid:
        if evidence_passages and len(evidence_passages) > 0:
            evid_blocks = []
            for idx, passage in enumerate(evidence_passages, 1):
                clean_p = passage.strip()
                evid_blocks.append(f"[Evidence {idx}]\n{clean_p}")
            sections.append("GOLD EVIDENCE:\n" + "\n\n".join(evid_blocks))
        else:
            sections.append("GOLD EVIDENCE:\nNo supporting evidence provided.")

    # 5. Auxiliary Reasoning (Opzionale / Ablation E)
    if (use_subquestion_reasoning or ablation == "E") and subquestion_reasoning:
        sections.append(f"REASONING CONTEXT:\n{subquestion_reasoning.strip()}")

    # 6. Current User Turn
    sections.append(f"CURRENT USER TURN:\n{query.strip()}")

    # 7. Final Task Trigger
    sections.append("TASK:\nProvide a direct, grounded answer to the current user turn.")

    return "\n\n".join(sections)


def build_refinement_prompt(
    query: str,
    conversation_history: str,
    evidence_passages: List[str],
    draft_answer: str,
    domain: Optional[str] = None,
) -> str:
    """Prompt per la strategia opzionale 'refine' a due stadi."""
    evid_text = ""
    for idx, passage in enumerate(evidence_passages, 1):
        evid_text += f"[Evidence {idx}] {passage.strip()}\n"

    return (
        "SYSTEM:\n"
        "You are an expert grounded editor. Your goal is to review and refine a draft answer.\n"
        "Check that all factual assertions are supported by the gold evidence.\n"
        "Remove any hallucinated or unsupported statements.\n"
        "Ensure conversational fluency and clear resolution of references.\n"
        "Do not include commentary or evaluation scores. Return only the final, corrected answer.\n\n"
        f"DOMAIN: {domain or 'General'}\n\n"
        f"CONVERSATION HISTORY:\n{conversation_history.strip()}\n\n"
        f"GOLD EVIDENCE:\n{evid_text.strip()}\n\n"
        f"CURRENT USER TURN: {query.strip()}\n\n"
        f"DRAFT ANSWER:\n{draft_answer.strip()}\n\n"
        "REFINED ANSWER:"
    )


def compute_generation_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Calcola le metriche di generazione offline (ROUGE-L, METEOR, BERTScore).
    Gestisce l'assenza delle librerie esterne con warning senza bloccare l'esecuzione.
    """
    metrics: Dict[str, float] = {}
    if not predictions or not references or len(predictions) != len(references):
        return metrics

    # 1. ROUGE-L
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l_scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        metrics["rougeL"] = float(np.mean(rouge_l_scores))
    except ImportError:
        logging.getLogger("subtrack_2b").warning(
            "Modulo 'rouge_score' non installato. Salto il calcolo di ROUGE-L."
        )

    # 2. METEOR
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)

        meteor_scores = [
            meteor_score([ref.split()], pred.split())
            for pred, ref in zip(predictions, references)
        ]
        metrics["meteor"] = float(np.mean(meteor_scores))
    except (ImportError, Exception) as e:
        logging.getLogger("subtrack_2b").warning(
            f"Modulo NLTK METEOR non disponibile o incompleto ({e}). Salto METEOR."
        )

    # 3. BERTScore
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            predictions,
            references,
            lang="en",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        metrics["bertscore_f1"] = float(F1.mean().item())
    except (ImportError, Exception) as e:
        logging.getLogger("subtrack_2b").warning(
            f"Libreria 'bert_score' non disponibile ({e}). Salto BERTScore."
        )

    return metrics


def score_grounding_lexical(
    answer: str,
    evidence_passages: List[str],
) -> float:
    """
    Euristica lessicale di grounding.

    Misura la frazione dei token di contenuto dell'answer che compaiono
    realmente come token nell'evidence.

    Non è una metrica ufficiale RETECO e deve essere usata solo come
    quality-control opzionale.
    """

    if not answer or not evidence_passages:
        return 0.0

    stop_words = {
        "the", "a", "an", "in", "on", "at", "to", "for",
        "of", "and", "or", "is", "are", "was", "were",
        "it", "this", "that", "with", "as", "by",
        "be", "been", "being", "from", "into", "than",
        "then", "they", "them", "their", "there",
        "which", "what", "when", "where", "who",
    }

    answer_tokens = [
        token.lower()
        for token in re.findall(
            r"\b\w+\b",
            answer,
        )
        if token.lower() not in stop_words
    ]

    if not answer_tokens:
        return 1.0

    evidence_tokens = {
        token.lower()
        for passage in evidence_passages
        for token in re.findall(
            r"\b\w+\b",
            passage,
        )
    }

    matches = sum(
        1
        for token in answer_tokens
        if token in evidence_tokens
    )

    return float(
        matches / len(answer_tokens)
    )


def select_best_candidate(
    candidates: List[str],
    query: str,
    history: str,
    evidence_passages: List[str],
    use_grounding: bool = True,
) -> str:
    """
    Seleziona il miglior candidato senza usare la risposta gold.

    Componenti:
    - grounding lessicale opzionale;
    - overlap con la query;
    - preferenza per risposte non vuote e di lunghezza ragionevole.

    Non utilizza mai reference answer.
    """

    if not candidates:
        return ""

    if len(candidates) == 1:
        return candidates[0].strip()

    def content_tokens(text: str) -> set:
        stop_words = {
            "the", "a", "an", "in", "on", "at", "to",
            "for", "of", "and", "or", "is", "are",
            "was", "were", "it", "this", "that",
            "with", "as", "by",
        }

        return {
            tok.lower()
            for tok in re.findall(
                r"\b\w+\b",
                text,
            )
            if tok.lower() not in stop_words
        }

    query_tokens = content_tokens(query)

    best_candidate = ""
    best_score = -float("inf")

    for candidate in candidates:
        clean = candidate.strip()

        if not clean:
            continue

        words = clean.split()
        word_count = len(words)

        # --------------------------------------------------------------
        # 1. Length quality
        # --------------------------------------------------------------
        if 10 <= word_count <= 250:
            length_score = 1.0
        elif 5 <= word_count < 10:
            length_score = 0.7
        elif 250 < word_count <= 350:
            length_score = 0.8
        else:
            length_score = 0.4

        # --------------------------------------------------------------
        # 2. Query relevance
        # --------------------------------------------------------------
        candidate_tokens = content_tokens(clean)

        if query_tokens and candidate_tokens:
            query_overlap = (
                len(query_tokens & candidate_tokens)
                / len(query_tokens)
            )
        else:
            query_overlap = 0.0

        # --------------------------------------------------------------
        # 3. Grounding
        # --------------------------------------------------------------
        if use_grounding:
            grounding = score_grounding_lexical(
                clean,
                evidence_passages,
            )
        else:
            grounding = 0.0

        # --------------------------------------------------------------
        # 4. Final score
        # --------------------------------------------------------------
        if use_grounding:
            score = (
                0.55 * grounding
                + 0.25 * query_overlap
                + 0.20 * length_score
            )
        else:
            score = (
                0.65 * query_overlap
                + 0.35 * length_score
            )

        if score > best_score:
            best_score = score
            best_candidate = clean

    # Se tutti i candidati sono vuoti, restituisci comunque il primo.
    if not best_candidate:
        return candidates[0].strip()

    return best_candidate