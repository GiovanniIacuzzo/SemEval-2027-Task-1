#!/usr/bin/env python3
"""
subtrack_2b/dataset/dataset.py

Modulo per il caricamento, la risoluzione dell'evidenza e la tokenizzazione
SFT per il Sub-track 2b di RETECO.

Caratteristiche:
- Lookup gold_doc_ids -> documents.jsonl con gestione dei documenti mancanti
- Supporto a gold_passages/evidence già presenti nel benchmark
- Flattening conversation -> turni individuali
- Split train/validation rigorosamente a livello di conversazione
- Domain-balanced temperature sampling deterministico
- Troncamento strutturato:
    query sempre preservata
    evidence prioritaria rispetto alla history
    history recente prioritaria rispetto alla history remota
- Supporto a max_history_turns
- Chat template del tokenizer quando disponibile
- Tokenizzazione SFT con label masking:
    prompt -> -100
    answer -> token IDs effettivi
"""

import json
import logging
import random
import re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from subtrack_2b.utils.utils import build_prompt, build_refinement_prompt, truncate_history

logger = logging.getLogger("subtrack_2b")


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class GroundedTurnSample:
    """Rappresentazione unificata di un turno conversazionale."""

    conversation_id: str
    domain: str
    turn_id: int
    topic_id: str
    query: str
    conversation_history: str
    gold_doc_ids: List[str]
    gold_passages: List[str]
    answer: Optional[str] = None
    original_query: Optional[str] = None
    original_answer: Optional[str] = None
    subquestion_reasoning: Optional[str] = None


# ==============================================================================
# CORPUS LOADING
# ==============================================================================

def load_corpus(documents_path: Union[str, Path]) -> Dict[str, str]:
    """
    Carica documents.jsonl e costruisce:

        doc_id -> document text

    Supporta sia:
        {"doc_id": "...", "content": "..."}
    sia:
        {"id": "...", "text": "..."}
    """
    path = Path(documents_path)

    if not path.exists():
        raise FileNotFoundError(f"Corpus non trovato in: {path}")

    corpus: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()

            if not line_str:
                continue

            try:
                item = json.loads(line_str)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "JSON non valido in %s alla riga %d: %s",
                    path,
                    line_no,
                    exc,
                )
                continue

            doc_id = str(
                item.get("doc_id")
                or item.get("id")
                or ""
            ).strip()

            content = (
                item.get("content")
                or item.get("text")
                or ""
            )

            content = str(content).strip()

            if doc_id:
                corpus[doc_id] = content

    logger.info(
        "Corpus caricato: %d documenti da %s",
        len(corpus),
        path,
    )

    return corpus


# ==============================================================================
# GOLD EVIDENCE RESOLUTION
# ==============================================================================

def _normalize_passage(value: Any) -> str:
    """
    Normalizza un passage proveniente da una struttura eventualmente eterogenea.

    Supporta:
    - string
    - {"content": "..."}
    - {"text": "..."}
    """
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        text = (
            value.get("content")
            or value.get("text")
            or value.get("passage")
            or ""
        )
        return str(text).strip()

    return str(value).strip()


def resolve_gold_passages(
    turn_record: Dict[str, Any],
    corpus: Dict[str, str],
    diagnostics: Dict[str, Any],
    dropped_gold_ids: frozenset[str] = frozenset(),
) -> List[str]:
    """
    Risolve il testo dell'evidenza gold.

    Priorità:
    1. gold_passages
    2. passages
    3. evidence
    4. gold_doc_ids / gold_ids -> corpus lookup

    Mantiene l'ordine dei passage / document IDs.
    """

    # --------------------------------------------------------------------------
    # 1. Evidenza già disponibile nel record
    # --------------------------------------------------------------------------
    direct_passages = (
        turn_record.get("gold_passages")
        or turn_record.get("passages")
        or turn_record.get("evidence")
    )

    if isinstance(direct_passages, list) and direct_passages:
        normalized = [
            _normalize_passage(p)
            for p in direct_passages
        ]

        normalized = [
            p for p in normalized
            if p
        ]

        if normalized:
            return normalized

    # --------------------------------------------------------------------------
    # 2. Lookup da gold_doc_ids
    # --------------------------------------------------------------------------
    gold_ids = (
        turn_record.get("gold_doc_ids")
        or turn_record.get("gold_ids")
        or []
    )

    if not isinstance(gold_ids, list):
        gold_ids = [gold_ids]

    passages: List[str] = []
    has_missing = False

    for doc_id in gold_ids:
        doc_id_str = str(doc_id).strip()

        if not doc_id_str:
            continue

        text = corpus.get(doc_id_str, "").strip()
        if text:
            passages.append(text)
            continue
        if doc_id_str in dropped_gold_ids:
            key = "documented_missing_gold_ids_count"
            diagnostics[key] = diagnostics.get(key, 0) + 1
            continue
        has_missing = True
        diagnostics["missing_gold_ids_count"] = diagnostics.get("missing_gold_ids_count", 0) + 1
        diagnostics.setdefault("missing_doc_ids_set", set()).add(doc_id_str)

    if has_missing:
        diagnostics["examples_with_missing_evidence"] = (
            diagnostics.get("examples_with_missing_evidence", 0) + 1
        )

    return passages


# ==============================================================================
# BENCHMARK LOADING
# ==============================================================================

def load_benchmark_records(
    benchmark_path: Union[str, Path]
) -> List[Dict[str, Any]]:
    """
    Carica benchmark_train/dev da:
    - JSON array
    - JSON object
    - JSON object con campo "conversations"
    - JSONL
    """

    path = Path(benchmark_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file non trovato: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # --------------------------------------------------------------------------
    # JSON array
    # --------------------------------------------------------------------------
    if content.startswith("["):
        try:
            parsed = json.loads(content)

            if not isinstance(parsed, list):
                raise ValueError(
                    f"Il file {path} inizia con '[' ma non contiene una lista."
                )

            return parsed

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON non valido in {path}: {exc}"
            ) from exc

    # --------------------------------------------------------------------------
    # JSON object
    # --------------------------------------------------------------------------
    if content.startswith("{"):
        try:
            parsed = json.loads(content)

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Struttura JSON inattesa in {path}."
                )

            if isinstance(parsed.get("conversations"), list):
                return parsed["conversations"]

            return [parsed]

        except json.JSONDecodeError:
            # Potrebbe trattarsi di JSONL.
            pass

    # --------------------------------------------------------------------------
    # JSONL fallback
    # --------------------------------------------------------------------------
    records: List[Dict[str, Any]] = []

    for line_no, line in enumerate(content.splitlines(), start=1):
        line_str = line.strip()

        if not line_str:
            continue

        try:
            item = json.loads(line_str)
        except json.JSONDecodeError as exc:
            logger.warning(
                "JSONL non valido in %s alla riga %d: %s",
                path,
                line_no,
                exc,
            )
            continue

        if isinstance(item, dict):
            records.append(item)

    return records


# ==============================================================================
# DOMAIN LOADING / FLATTENING
# ==============================================================================

def load_domain_turn_samples(
    benchmark_path: Union[str, Path],
    documents_path: Union[str, Path],
    domain: str,
    diagnostics: Dict[str, Any],
    dropped_gold_ids: frozenset[str] = frozenset(),
) -> List[GroundedTurnSample]:
    """
    Carica un dominio completo e converte:

        conversation -> individual turns
    """

    corpus = load_corpus(documents_path) if Path(documents_path).is_file() else {}
    conversations = load_benchmark_records(benchmark_path)

    turn_samples: List[GroundedTurnSample] = []

    for conv_idx, conv in enumerate(conversations):
        conv_id = str(
            conv.get("id")
            or conv.get("conversation_id")
            or f"{domain}_conv_{conv_idx}"
        ).strip()

        orig_query = conv.get("original_query")
        orig_ans = conv.get("original_answer")

        turns = conv.get("turns", [])

        if not isinstance(turns, list):
            logger.warning(
                "Conversazione %s nel dominio %s ha turns non-list. Ignorata.",
                conv_id,
                domain,
            )
            continue

        for turn_idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue

            try:
                turn_id = int(
                    turn.get("turn_id", turn_idx + 1)
                )
            except (TypeError, ValueError):
                turn_id = turn_idx + 1

            topic_id = f"{conv_id}_turn_{turn_id}"

            query = str(
                turn.get("query", "")
            ).strip()

            if not query:
                diagnostics["empty_query_count"] = (
                    diagnostics.get("empty_query_count", 0) + 1
                )

            answer = turn.get("answer")

            if answer is not None:
                answer = str(answer).strip()

            if not answer:
                diagnostics["empty_answer_count"] = (
                    diagnostics.get("empty_answer_count", 0) + 1
                )

            history = str(
                turn.get(
                    "conversation_history",
                    "No previous conversation.",
                )
            ).strip()

            sub_reasoning = turn.get(
                "subquestion_reasoning"
            )

            if sub_reasoning is not None:
                sub_reasoning = str(
                    sub_reasoning
                ).strip()

            gold_ids_raw = (
                turn.get("gold_doc_ids")
                or turn.get("gold_ids")
                or []
            )

            if not isinstance(gold_ids_raw, list):
                gold_ids_raw = [gold_ids_raw]

            gold_ids = [
                str(x).strip()
                for x in gold_ids_raw
                if str(x).strip()
            ]

            passages = resolve_gold_passages(turn, corpus, diagnostics, dropped_gold_ids)

            if not passages:
                diagnostics["examples_without_evidence"] = (
                    diagnostics.get("examples_without_evidence", 0) + 1
                )

            sample = GroundedTurnSample(
                conversation_id=conv_id,
                domain=domain,
                turn_id=turn_id,
                topic_id=topic_id,
                query=query,
                conversation_history=history,
                gold_doc_ids=gold_ids,
                gold_passages=passages,
                answer=answer,
                original_query=(
                    str(orig_query).strip()
                    if orig_query is not None
                    else None
                ),
                original_answer=(
                    str(orig_ans).strip()
                    if orig_ans is not None
                    else None
                ),
                subquestion_reasoning=sub_reasoning,
            )

            turn_samples.append(sample)

    diagnostics["num_conversations_loaded"] = (
        diagnostics.get("num_conversations_loaded", 0)
        + len(conversations)
    )

    return turn_samples


def _normalize_active_domains(
    active_domains: Union[str, List[str], Tuple[str, ...]],
    all_domains: List[str],
) -> List[str]:
    """
    Normalizza active_domains evitando il bug classico:

        active_domains = "drones"

    che altrimenti diventerebbe:

        ["d", "r", "o", "n", "e", "s"]

    Restituisce sempre una lista di domini.
    """

    if isinstance(active_domains, str):
        value = active_domains.strip()

        if value.lower() == "all":
            return list(all_domains)

        return [value]

    if isinstance(active_domains, (list, tuple)):
        normalized = [
            str(x).strip()
            for x in active_domains
            if str(x).strip()
        ]

        if len(normalized) == 1 and normalized[0].lower() == "all":
            return list(all_domains)

        return normalized

    raise TypeError(
        "active_domains deve essere una stringa o una lista/tupla di stringhe; "
        f"ricevuto: {type(active_domains)}"
    )


def load_all_domains_data(
    base_dir: Union[str, Path],
    active_domains: Union[str, List[str]],
    all_domains: List[str],
    split_filename: str,
    documents_filename: str = "documents.jsonl",
    require_corpus: bool = True,
) -> Tuple[List[GroundedTurnSample], Dict[str, Any]]:
    """
    Carica tutti i domini selezionati.
    """

    base_path = Path(base_dir)
    manifest_path = base_path.parent / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() and require_corpus else {}

    domains_to_load = _normalize_active_domains(
        active_domains,
        all_domains,
    )

    all_samples: List[GroundedTurnSample] = []

    diagnostics: Dict[str, Any] = {
        "missing_gold_ids_count": 0,
        "documented_missing_gold_ids_count": 0,
        "examples_with_missing_evidence": 0,
        "examples_without_evidence": 0,
        "empty_query_count": 0,
        "empty_answer_count": 0,
        "missing_doc_ids_set": set(),
        "domains_loaded": [],
        "num_conversations_loaded": 0,
    }

    for domain in domains_to_load:
        domain_dir = base_path / domain

        benchmark_file = domain_dir / split_filename
        documents_file = domain_dir / documents_filename

        if not benchmark_file.exists():
            logger.warning(
                "Benchmark mancante per il dominio '%s': %s",
                domain,
                benchmark_file,
            )
            continue

        if require_corpus and not documents_file.exists():
            logger.warning(
                "Corpus mancante per il dominio '%s': %s",
                domain,
                documents_file,
            )
            continue

        samples = load_domain_turn_samples(
            benchmark_path=benchmark_file,
            documents_path=documents_file,
            domain=domain,
            diagnostics=diagnostics,
            dropped_gold_ids=frozenset(manifest.get("track2_recor", {}).get(domain, {}).get("dropped_gold_ids", [])),
        )

        all_samples.extend(samples)

        diagnostics["domains_loaded"].append(domain)

        logger.info(
            "Dominio [%-18s]: %d turni caricati.",
            domain,
            len(samples),
        )

    logger.info(
        "Dataset totale: %d turni | domini caricati: %s",
        len(all_samples),
        diagnostics["domains_loaded"],
    )

    return all_samples, diagnostics


# ==============================================================================
# CONVERSATION-LEVEL TRAIN / VALIDATION SPLIT
# ==============================================================================

def split_conversations_train_val(
    samples: List[GroundedTurnSample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[GroundedTurnSample], List[GroundedTurnSample]]:
    """
    Divide il dataset a livello di CONVERSAZIONE.

    Nessuna conversazione può comparire contemporaneamente in:
        train
        internal validation
    """

    if not samples:
        return [], []

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            f"val_ratio deve essere compreso tra 0 e 1; ricevuto {val_ratio}"
        )

    conversations_by_domain = defaultdict(set)
    for sample in samples:
        conversations_by_domain[sample.domain].add(sample.conversation_id)

    rng = random.Random(seed)
    validation = set()
    for domain, conversations in sorted(conversations_by_domain.items()):
        ids = sorted(conversations)
        if len(ids) < 2:
            logger.warning("Dominio %s senza conversazioni sufficienti per la validation.", domain)
            continue
        rng.shuffle(ids)
        count = min(len(ids) - 1, max(1, round(len(ids) * val_ratio)))
        validation.update((domain, conv_id) for conv_id in ids[:count])

    train_samples = []
    val_samples = []
    for sample in samples:
        partition = val_samples if (sample.domain, sample.conversation_id) in validation else train_samples
        partition.append(sample)

    return train_samples, val_samples


# ==============================================================================
# DOMAIN-BALANCED SAMPLING
# ==============================================================================

def apply_domain_balanced_sampling(
    samples: List[GroundedTurnSample],
    temperature: float = 0.5,
    seed: int = 42,
) -> List[GroundedTurnSample]:
    """
    Applica temperature sampling ai domini.

    IMPORTANTE:
        P(domain) ∝ N_domain ** temperature

    quindi:
        temperature = 1.0 -> distribuzione originale
        temperature < 1.0   -> smoothing / maggiore bilanciamento
        temperature -> 0    -> distribuzione quasi uniforme

    Il precedente utilizzo di 1 / temperature produceva invece l'effetto
    opposto quando temperature < 1.
    """

    if not samples:
        return []

    if temperature <= 0.0:
        raise ValueError(
            f"temperature deve essere > 0; ricevuto {temperature}"
        )

    domain_groups: Dict[str, List[GroundedTurnSample]] = {}

    for sample in samples:
        domain_groups.setdefault(
            sample.domain,
            [],
        ).append(sample)

    counts = {
        domain: len(items)
        for domain, items in domain_groups.items()
    }

    total_samples = len(samples)

    # N_d ^ temperature
    scaled = {
        domain: float(count) ** temperature
        for domain, count in counts.items()
    }

    sum_scaled = sum(scaled.values())

    if sum_scaled <= 0:
        return list(samples)

    raw_targets = {
        domain: (scaled[domain] / sum_scaled) * total_samples
        for domain in scaled
    }

    # Floor iniziale
    target_counts = {
        domain: max(
            1,
            int(np.floor(value)),
        )
        for domain, value in raw_targets.items()
    }

    # Correzione per far sì che la somma sia esattamente total_samples
    current_total = sum(target_counts.values())

    if current_total < total_samples:
        remaining = total_samples - current_total

        ordered_domains = sorted(
            raw_targets.keys(),
            key=lambda d: (
                raw_targets[d] - np.floor(raw_targets[d]),
                raw_targets[d],
            ),
            reverse=True,
        )

        index = 0

        while remaining > 0:
            domain = ordered_domains[index % len(ordered_domains)]
            target_counts[domain] += 1
            remaining -= 1
            index += 1

    elif current_total > total_samples:
        excess = current_total - total_samples

        ordered_domains = sorted(
            target_counts.keys(),
            key=lambda d: (
                raw_targets[d] - np.floor(raw_targets[d]),
                raw_targets[d],
            ),
        )

        index = 0

        while excess > 0 and ordered_domains:
            domain = ordered_domains[
                index % len(ordered_domains)
            ]

            if target_counts[domain] > 1:
                target_counts[domain] -= 1
                excess -= 1

            index += 1

    rng = random.Random(seed)

    balanced_samples: List[GroundedTurnSample] = []

    for domain, items in domain_groups.items():
        needed = target_counts[domain]

        if needed <= len(items):
            chosen = rng.sample(
                items,
                needed,
            )
        else:
            # Oversampling controllato quando il dominio è piccolo
            chosen = list(items)

            chosen.extend(
                rng.choices(
                    items,
                    k=needed - len(items),
                )
            )

        balanced_samples.extend(chosen)

    rng.shuffle(balanced_samples)

    logger.info(
        "Domain sampling: originale=%s | target=%s",
        counts,
        target_counts,
    )

    return balanced_samples


# ==============================================================================
# TEXT / TOKEN BUDGET HELPERS
# ==============================================================================

def _truncate_history_by_turns(
    history: str,
    max_history_turns: Optional[int],
) -> str:
    """
    Troncamento best-effort della history a livello di turni.

    Supporta separatori comuni:
    - blocchi separati da righe vuote
    - Turn N:
    - User:
    - Assistant:
    - Human:
    - AI:
    - Customer:
    - Agent:
    - Bot:

    Se la struttura non è riconoscibile, restituisce la history originale.
    La successiva token truncation rimane comunque attiva.
    """

    if not history:
        return history

    if not max_history_turns or max_history_turns <= 0:
        return history

    text = history.strip()

    # Caso 1: blocchi separati da righe vuote
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n+", text)
        if block.strip()
    ]

    if len(blocks) > max_history_turns:
        return "\n\n".join(
            blocks[-max_history_turns:]
        )

    # Caso 2: prefissi strutturati
    pattern = re.compile(
        r"(?=^\s*"
        r"(?:Turn\s+\d+|User|Assistant|Human|AI|Customer|Agent|Bot)"
        r"\s*:)",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    structured_blocks = [
        block.strip()
        for block in pattern.split(text)
        if block.strip()
    ]

    if len(structured_blocks) > max_history_turns:
        return "\n".join(
            structured_blocks[-max_history_turns:]
        )

    return text


def _tokenize_ids(
    tokenizer: Any,
    text: str,
) -> List[int]:
    """
    Tokenizzazione semplice senza chat template.
    """
    return tokenizer.encode(
        text,
        add_special_tokens=False,
    )


def _truncate_passages_to_budget(
    passages: List[str],
    tokenizer: Any,
    max_tokens: int,
    focus: str = "",
) -> List[str]:
    """
    Mantiene tutti i passage, quando possibile, distribuendo il budget
    proporzionalmente alla lunghezza.

    In questo modo, se l'evidence è troppo lunga:
    - non si elimina semplicemente tutto ciò che viene dopo il primo passage;
    - ogni gold passage conserva una porzione rappresentativa.
    """

    if not passages:
        return []

    if max_tokens <= 0:
        return []

    tokenized = [
        _tokenize_ids(tokenizer, passage)
        for passage in passages
    ]

    total_tokens = sum(
        len(tokens)
        for tokens in tokenized
    )

    if total_tokens <= max_tokens:
        return list(passages)

    n = len(passages)

    if max_tokens < n:
        # Budget estremo: assegna almeno un token ai primi passage
        # fino a esaurimento.
        truncated: List[str] = []

        for tokens in tokenized[:max_tokens]:
            if tokens:
                truncated.append(
                    tokenizer.decode(
                        tokens[:1],
                        skip_special_tokens=True,
                    ).strip()
                )

        return [
            p for p in truncated if p
        ]

    # Allocazione proporzionale iniziale
    allocations = [
        max(
            1,
            int(
                round(
                    max_tokens
                    * len(tokens)
                    / max(total_tokens, 1)
                )
            ),
        )
        for tokens in tokenized
    ]

    # Correggi eventuali differenze dovute agli arrotondamenti
    current = sum(allocations)

    while current > max_tokens:
        idx = max(
            range(n),
            key=lambda i: allocations[i],
        )

        if allocations[idx] <= 1:
            break

        allocations[idx] -= 1
        current -= 1

    while current < max_tokens:
        idx = max(
            range(n),
            key=lambda i: len(tokenized[i]) - allocations[i],
        )

        if allocations[idx] >= len(tokenized[idx]):
            break

        allocations[idx] += 1
        current += 1

    truncated_passages: List[str] = []

    focus_words = set(re.findall(r"\b\w{3,}\b", focus.lower()))
    for passage, tokens, allocation in zip(passages, tokenized, allocations):
        if not tokens:
            continue

        if len(tokens) <= allocation:
            truncated_passages.append(passage)
            continue
        if focus_words:
            sentences = re.split(r"(?<=[.!?])\s+|\n+", passage)
            order = sorted(
                range(len(sentences)),
                key=lambda i: -len(focus_words & set(re.findall(r"\b\w{3,}\b", sentences[i].lower()))),
            )
            selected = []
            for i in order:
                candidate = sorted([*selected, i])
                text = " ".join(sentences[j] for j in candidate)
                if len(_tokenize_ids(tokenizer, text)) <= allocation:
                    selected = candidate
            if selected:
                truncated_passages.append(" ".join(sentences[i] for i in selected))
                continue
        clipped = tokens[:allocation]

        text = tokenizer.decode(
            clipped,
            skip_special_tokens=True,
        ).strip()

        if text:
            truncated_passages.append(text)

    return truncated_passages


# ==============================================================================
# CHAT TEMPLATE HELPERS
# ==============================================================================

def _encode_prompt_with_chat_template(tokenizer: Any, prompt_text: str) -> List[int]:
    """Encode the complete chat prompt once for both training and generation."""
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    else:
        encoded = tokenizer.encode(prompt_text, add_special_tokens=True)
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not encoded or not all(isinstance(token, int) for token in encoded):
        raise TypeError("Il tokenizer non ha restituito una sequenza di ID interi.")
    return list(encoded)


# ==============================================================================
# SFT DATASET
# ==============================================================================

class GroundedGenerationDataset(Dataset):
    """
    Dataset PyTorch per SFT generativo grounded.

    Durante il training:
        input_ids = prompt + answer
        labels    = -100 sul prompt + answer IDs

    Durante inference:
        restituisce prompt_ids / prompt_text / messages,
        senza gold answer nei dati passati al modello.
    """

    def __init__(
        self,
        samples: List[GroundedTurnSample],
        tokenizer: Any,
        max_input_tokens: int = 3584,
        max_output_tokens: int = 512,
        max_history_turns: Optional[int] = None,
        ablation: str = "D",
        include_domain: bool = True,
        use_subquestion_reasoning: bool = False,
        is_training: bool = True,
        evidence_strategy: str = "adaptive",
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer

        self.max_input_tokens = int(
            max_input_tokens
        )

        self.max_output_tokens = int(
            max_output_tokens
        )

        self.max_history_turns = (
            int(max_history_turns)
            if max_history_turns is not None
            else None
        )

        self.ablation = ablation
        self.include_domain = include_domain
        self.use_subquestion_reasoning = (
            use_subquestion_reasoning
        )
        self.is_training = is_training

        if evidence_strategy not in {"prefix", "adaptive", "sentences"}:
            raise ValueError(f"Strategia evidenza sconosciuta: {evidence_strategy}")
        if min(self.max_input_tokens, self.max_output_tokens) <= 0:
            raise ValueError("I limiti dei token devono essere positivi.")
        self.evidence_strategy = evidence_strategy

    def __len__(self) -> int:
        return len(self.samples)

    def prepare_prompt(
        self, sample: GroundedTurnSample, draft_answer: Optional[str] = None
    ) -> Tuple[str, List[int], Dict[str, Any]]:
        use_history = self.ablation in {"B", "D", "E"} or draft_answer is not None
        use_evidence = self.ablation in {"C", "D", "E"} or draft_answer is not None
        original_history = sample.conversation_history if use_history else ""
        history = _truncate_history_by_turns(original_history, self.max_history_turns)
        history_limit = self.max_input_tokens // 4
        history, _ = truncate_history(history, history_limit, self.tokenizer)
        passages = sample.gold_passages if use_evidence else []

        def render(evidence):
            if draft_answer is not None:
                prompt = build_refinement_prompt(
                    sample.query, history, evidence, draft_answer, sample.domain
                )
            else:
                prompt = build_prompt(
                    query=sample.query,
                    conversation_history=history,
                    evidence_passages=evidence,
                    domain=sample.domain,
                    subquestion_reasoning=sample.subquestion_reasoning,
                    ablation=self.ablation,
                    include_domain=self.include_domain,
                    use_subquestion_reasoning=self.use_subquestion_reasoning,
                )
            return prompt, _encode_prompt_with_chat_template(self.tokenizer, prompt)

        _, base_ids = render([])
        while len(base_ids) + bool(passages) * 32 > self.max_input_tokens and history_limit:
            history_limit //= 2
            history, _ = truncate_history(history, history_limit, self.tokenizer)
            _, base_ids = render([])
        if len(base_ids) > self.max_input_tokens:
            raise ValueError(f"{sample.topic_id}: domanda, istruzioni o bozza oltre il budget del prompt.")

        budget = self.max_input_tokens - len(base_ids)
        if self.evidence_strategy == "prefix":
            budget = min(budget, self.max_input_tokens // 2)
        focus = f"{sample.query} {history}" if self.evidence_strategy == "sentences" else ""
        evidence = _truncate_passages_to_budget(passages, self.tokenizer, budget, focus)
        prompt, ids = render(evidence)
        while len(ids) > self.max_input_tokens and budget:
            budget = max(0, budget - (len(ids) - self.max_input_tokens) - 1)
            evidence = _truncate_passages_to_budget(passages, self.tokenizer, budget, focus)
            prompt, ids = render(evidence)
        if passages and not evidence:
            raise ValueError(f"{sample.topic_id}: spazio insufficiente per i passaggi gold.")
        if self.is_training:
            return prompt, ids, {}
        diagnostics = {
            "input_tokens": len(ids),
            "history_truncated": use_history and history != original_history,
            "evidence_truncated": evidence != passages,
            "passages_used": len(evidence),
            "evidence_tokens": sum(len(_tokenize_ids(self.tokenizer, text)) for text in evidence),
            "original_evidence_tokens": sum(len(_tokenize_ids(self.tokenizer, text)) for text in passages),
            "evidence_shown": evidence,
            "history_shown": history,
        }
        return prompt, ids, diagnostics

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        sample = self.samples[idx]

        prompt_text, prompt_ids, diagnostics = self.prepare_prompt(
            sample
        )

        # ----------------------------------------------------------------------
        # Inference dataset
        # ----------------------------------------------------------------------
        if not self.is_training:
            return {
                "prompt_ids": prompt_ids,
                "prompt_text": prompt_text,
                "prompt_diagnostics": diagnostics,
                "sample": sample,
            }

        # ----------------------------------------------------------------------
        # Target answer
        # ----------------------------------------------------------------------
        answer_text = (
            sample.answer.strip()
            if sample.answer
            else ""
        )

        if not answer_text:
            logger.warning(
                "Answer vuota per sample %s. "
                "Verrà utilizzato solo EOS come target.",
                sample.topic_id,
            )

        answer_ids = self.tokenizer.encode(
            answer_text,
            add_special_tokens=False,
        )

        eos_id = self.tokenizer.eos_token_id

        if eos_id is not None:
            if not answer_ids or answer_ids[-1] != eos_id:
                answer_ids.append(eos_id)

        # ----------------------------------------------------------------------
        # Output truncation
        # ----------------------------------------------------------------------
        if len(answer_ids) > self.max_output_tokens:
            answer_ids = answer_ids[
                :self.max_output_tokens
            ]

            # Mantieni EOS come ultimo token quando possibile.
            if (
                eos_id is not None
                and answer_ids
                and answer_ids[-1] != eos_id
            ):
                answer_ids[-1] = eos_id

        # ----------------------------------------------------------------------
        # Causal SFT:
        #
        # prompt -> label -100
        # answer -> target token IDs
        # ----------------------------------------------------------------------
        input_ids = prompt_ids + answer_ids

        labels = (
            [-100] * len(prompt_ids)
            + answer_ids
        )

        attention_mask = [1] * len(input_ids)
        if not all(
            isinstance(x, int)
            for x in input_ids
        ):
            bad_items = [
                (i, type(x).__name__, repr(x)[:100])
                for i, x in enumerate(input_ids)
                if not isinstance(x, int)
            ]

            raise TypeError(
                f"Tokenizzazione non valida per sample {sample.topic_id}. "
                f"Trovati token non interi: {bad_items[:10]}"
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
            "sample": sample,
            "prompt_text": prompt_text,
        }


# ==============================================================================
# COLLATOR
# ==============================================================================

class GroundedDataCollator:
    """
    Data collator con dynamic padding.

    Il padding:
        input_ids -> pad_token_id
        attention_mask -> 0
        labels -> -100
    """

    def __init__(
        self,
        pad_token_id: int,
    ) -> None:
        self.pad_token_id = int(
            pad_token_id
        )

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        input_ids = [
            item["input_ids"]
            for item in batch
        ]

        attention_mask = [
            item["attention_mask"]
            for item in batch
        ]

        labels = [
            item["labels"]
            for item in batch
        ]

        max_len = max(
            len(ids)
            for ids in input_ids
        )

        padded_input_ids: List[torch.Tensor] = []
        padded_attention_mask: List[torch.Tensor] = []
        padded_labels: List[torch.Tensor] = []

        for i_ids, a_mask, lbls in zip(
            input_ids,
            attention_mask,
            labels,
        ):
            pad_len = max_len - len(i_ids)

            if pad_len > 0:
                padded_input_ids.append(
                    torch.cat(
                        [
                            i_ids,
                            torch.full(
                                (pad_len,),
                                self.pad_token_id,
                                dtype=torch.long,
                            ),
                        ]
                    )
                )

                padded_attention_mask.append(
                    torch.cat(
                        [
                            a_mask,
                            torch.zeros(
                                (pad_len,),
                                dtype=torch.long,
                            ),
                        ]
                    )
                )

                padded_labels.append(
                    torch.cat(
                        [
                            lbls,
                            torch.full(
                                (pad_len,),
                                -100,
                                dtype=torch.long,
                            ),
                        ]
                    )
                )
            else:
                padded_input_ids.append(i_ids)
                padded_attention_mask.append(a_mask)
                padded_labels.append(lbls)

        return {
            "input_ids": torch.stack(
                padded_input_ids
            ),
            "attention_mask": torch.stack(
                padded_attention_mask
            ),
            "labels": torch.stack(
                padded_labels
            ),
        }