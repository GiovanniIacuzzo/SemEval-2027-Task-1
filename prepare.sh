#!/usr/bin/env bash
# ==============================================================================
# prepare.sh
# Inizializzazione e verifica dell'ambiente di lavoro per SEMEVAL-2027 RETECO.
# ==============================================================================

set -euo pipefail

# Colori per il terminale
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   SEMEVAL-2027 TASK 1 (RETECO) - SETUP AMBIENTE DI LAVORO       ${NC}"
echo -e "${BLUE}================================================================${NC}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ------------------------------------------------------------------------------
# 1. Creazione della Struttura di Directory
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/5] Creazione struttura cartelle di progetto...${NC}"

DIRECTORIES=(
    "data/sample/track2_recor"
    "data/reteco_data/track2_recor"
    "checkpoints/subtrack_2a"
    "checkpoints/subtrack_2b"
    "outputs/subtrack_2a/logs"
    "outputs/subtrack_2b/logs"
)

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "  [+] Creata cartella: ${GREEN}$dir${NC}"
    else
        echo -e "  [.] Esistente: $dir"
    fi
done

# ------------------------------------------------------------------------------
# 2. Configurazione Permessi di Esecuzione
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/5] Impostazione permessi sugli script...${NC}"
chmod +x download_data.py prepare.sh 2>/dev/null || true
if [ -f "subtrack_2a/train.py" ]; then
    chmod +x subtrack_2a/train.py subtrack_2a/inference.py subtrack_2a/generate_submission.py 2>/dev/null || true
fi
echo -e "  ${GREEN}✓${NC} Permessi di esecuzione aggiornati."

# ------------------------------------------------------------------------------
# 3. Download Starter Kit Ufficiale (se assente)
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/5] Verifica starter kit ufficiale...${NC}"
if [ ! -d "starter_kit" ] || [ ! -f "starter_kit/official_baseline.py" ]; then
    echo -e "  Starter kit non trovato. Download da DataScienceUIBK/RETECO..."
    TMP_CLONE_DIR=$(mktemp -d)
    git clone --depth 1 https://github.com/DataScienceUIBK/RETECO.git "$TMP_CLONE_DIR" 2>/dev/null || true
    
    if [ -d "$TMP_CLONE_DIR/starter_kit" ]; then
        mkdir -p starter_kit
        cp -r "$TMP_CLONE_DIR/starter_kit/"* starter_kit/
        echo -e "  ${GREEN}✓${NC} Starter kit importato con successo in starter_kit/"
    else
        echo -e "  ${YELLOW}!${NC} Impossibile clonare lo starter kit (connessione assente o repository privato)."
    fi
    rm -rf "$TMP_CLONE_DIR"
else
    echo -e "  ${GREEN}✓${NC} Starter kit ufficiale già presente."
fi

# ------------------------------------------------------------------------------
# 4. Controllo / Installazione Ambiente Conda
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/5] Controllo ambiente Conda 'semeval'...${NC}"

if command -v conda &> /dev/null; then
    ENV_EXISTS=$(conda env list | grep -w "semeval" || true)
    if [ -z "$ENV_EXISTS" ]; then
        echo -e "  L'ambiente Conda 'semeval' non esiste. Creazione da environment.yaml in corso..."
        conda env create -f environment.yaml
        echo -e "  ${GREEN}✓${NC} Ambiente 'semeval' creato con successo."
    else
        echo -e "  ${GREEN}✓${NC} Ambiente 'semeval' già presente."
        echo -e "  (Per aggiornarlo con nuove dipendenze: conda env update -f environment.yaml --prune)"
    fi
else
    echo -e "  ${RED}!${NC} Conda non rilevato nel PATH di sistema."
    echo -e "    Assicurati di installare Miniconda/Anaconda oppure procedi tramite virtualenv standard:"
    echo -e "    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

# ------------------------------------------------------------------------------
# 5. Diagnostica Hardware e Test di Compatibilità
# ------------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/5] Esecuzione diagnostica hardware e librerie di runtime...${NC}"

# Esegue un probe diagnostico inline con Python
python3 - << 'EOF'
import sys
import platform

print(f"  Sistema Operativo: {platform.system()} {platform.release()} ({platform.machine()})")
print(f"  Versione Python:   {sys.version.split()[0]}")

# Test PyTorch e acceleratori
try:
    import torch
    print(f"  PyTorch:           v{torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Acceleratore:      CUDA [GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB]")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"  Acceleratore:      Apple Silicon MPS (Metal Performance Shaders)")
    else:
        print(f"  Acceleratore:      CPU Standard")
except ImportError:
    print("  PyTorch:           NON INSTALLATO")

# Test metrica ufficiale e Hugging Face
try:
    import pytrec_eval
    print("  pytrec_eval:       INSTALLATO (Metrica ufficiale ndcg_cut_10 attiva)")
except ImportError:
    print("  pytrec_eval:       NON INSTALLATO (Verificare ambiente conda)")

try:
    import transformers
    print(f"  Transformers:      v{transformers.__version__}")
except ImportError:
    print("  Transformers:      NON INSTALLATO")
EOF

echo -e "\n${BLUE}================================================================${NC}"
echo -e "${GREEN}✓ Preparazione completata con successo!${NC}"
echo -e "Per iniziare a lavorare:"
echo -e "  1. Attiva l'ambiente: ${YELLOW}conda activate reteco${NC}"
echo -e "  2. Scarica i dati di prova (su Mac): ${YELLOW}python download_data.py --sample${NC}"
echo -e "     oppure i dati completi (su Lightning AI): ${YELLOW}python download_data.py --full${NC}"
echo -e "${BLUE}================================================================${NC}\n"