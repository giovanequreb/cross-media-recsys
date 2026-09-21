# CLAUDE.md — Cross-Media Recommender

File condiviso tra Claude Code (lavora sul repo) e Claude chat (spiega, decide l'architettura). Adam fa da ponte: copia le sezioni aggiornate da una parte all'altra.

## Chi sono

Adam, 22 anni, sviluppatore full stack (JS/TS, C++, C#, SQL, PHP), vivo a Lucca. Non conosco ancora Python. Uso macOS.

## Obiettivo

Costruire un progetto AI da portfolio GitHub per crescere verso ruoli di AI applicata / solutions engineer / freelance, anche da remoto o all'estero.

## Il progetto

Sistema di raccomandazione cross-media: consiglia musica in base ai gusti su film e serie. Ispirato all'app Podiums (valutazione tramite confronti a coppie).

Piano in tre livelli:

1. Prototipo senza training: descrizioni generate da LLM + embeddings, raccomandazioni per similarità del coseno. (Locale, gratis: `sentence-transformers`, modello `all-MiniLM-L6-v2`.)
2. Modello allenato su dataset cross-domain (Douban, Amazon Reviews film + CD).
3. Confronti a coppie (Elo / Bradley-Terry) e app web completa.

## Regole di lavoro (per Claude Code)

- Procedi un passo alla volta. Non implementare più passi del richiesto.
- Spiega ogni riga di codice Python in italiano, con paragoni a JS/TS quando aiutano (venv ≈ node_modules, pip ≈ npm, requirements.txt ≈ lockfile, type hint ≈ TypeScript).
- Verifica librerie e API aggiornate prima di usarle (es. restrizioni API Spotify): non fidarti della memoria, controlla la documentazione.
- Tieni il repo adatto a un portfolio: README chiaro, struttura pulita, `requirements.txt` aggiornato, commit piccoli con messaggi sensati, niente segreti nel repo (`.env` nel `.gitignore`).
- Se una scelta è ambigua, chiedi invece di decidere in silenzio.
- A fine di ogni passo, aggiorna la sezione "Stato attuale" qui sotto.

## Struttura del repo

```
cross-media-recsys/
├── data/            # film, serie, brani
├── src/             # codice del progetto
├── notebooks/       # esperimenti esplorativi
├── requirements.txt
├── README.md
├── CLAUDE.md
└── .gitignore
```

## Stato attuale (aggiornato da Claude Code)

Da Claude Code → chat. Adam incolla questa sezione nella chat.

- Livello: 1 (prototipo senza training)
- Passo corrente: 1 — ambiente e struttura del progetto (**completato**); si passa al Passo 2
- Fatto:
  - [x] Python 3.12 installato
  - [x] venv `.venv` creato e attivato
  - [x] dipendenze installate (`sentence-transformers`, `numpy`, `pandas`)
  - [x] `requirements.txt` generato
  - [x] struttura cartelle + `.gitignore` + `git init`
  - [x] smoke test `src/smoke_test.py` eseguito (output: shape `(3, 384)`)
- Versioni installate: Python 3.12.3, `sentence-transformers` 6.1.0, `torch` 2.14.0, `numpy` 2.5.3, `pandas` 3.0.6, `scikit-learn` 1.9.1 (lista completa in `requirements.txt`)
- Problemi / errori incontrati:
  - Nessun errore. Al primo avvio compare un warning su richieste non autenticate a Hugging Face Hub: innocuo, si può ignorare (serve `HF_TOKEN` solo per limiti più alti).
  - In `sentence-transformers` 6.x il primo parametro di `encode` si chiama `inputs` (nelle versioni vecchie era `sentence`/`sentences`): verificato sul pacchetto installato. Lo passiamo in modo posizionale.
- Decisioni prese: modello di embedding locale `all-MiniLM-L6-v2`; descrizioni in inglese; README in inglese (repo da portfolio); `.gitkeep` in `data/` e `notebooks/` perché git non traccia cartelle vuote.

## Prossimo passo (scritto dalla chat)

Da chat → Claude Code. Adam incolla qui il blocco che ricevi dalla chat.

- Passo 2: dataset minimo (~20 film/serie e ~50 brani con metadati di base).
- Dettagli da definire nella chat, ora che il Passo 1 è completato. Domande aperte: formato (CSV o JSON), campi dei metadati, chi sceglie i titoli, se le descrizioni in inglese vanno nel dataset o in un passo separato.

## Log

Una riga per passo completato: data, cosa, commit.

- 2026-09-21 — Passo 1: ambiente, struttura, `.gitignore`, `requirements.txt`, smoke test `(3, 384)` — commit `56d5f64` (setup), `45b0a35` (smoke test); README e CLAUDE.md nel commit docs successivo.
