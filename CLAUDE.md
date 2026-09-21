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
├── data/            # titles.csv (film/serie), tracks.csv (brani), raw/ (download grezzi, ignorata da git)
├── src/             # smoke_test.py, fetch_titles.py, build_tracks.py
├── notebooks/       # esperimenti esplorativi
├── requirements.txt
├── .env.example     # modello per la chiave TMDB (.env è ignorato da git)
├── README.md
├── CLAUDE.md
└── .gitignore
```

## Stato attuale (aggiornato da Claude Code)

Da Claude Code → chat. Adam incolla questa sezione nella chat.

- Livello: 1 (prototipo senza training)
- Passo corrente: 2 — dataset minimo (**completato**). Il Passo 1 (ambiente, struttura, smoke test `(3, 384)`) è chiuso: vedi Log.
- Fatto (Passo 2):
  - [x] `src/fetch_titles.py`: 20 film/serie da TMDB → `data/titles.csv` (colonne: `tmdb_id`, `title`, `year`, `type` = `movie`/`tv`, `genres` separati da `|`, `overview` in inglese)
  - [x] `src/build_tracks.py`: 50 brani scelti a mano dal dataset Hugging Face `maharshipandya/spotify-tracks-dataset` → `data/tracks.csv` (colonne: `track_id`, `title`, `artist`, `album`, `genre_label`, `popularity`, `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`)
  - [x] `.env` con la chiave TMDB (ignorato da git, permessi 600) + `.env.example` committato
  - [x] README aggiornato: sezione Dati, credit TMDB, struttura, roadmap
- Versioni installate: Python 3.12.3, `sentence-transformers` 6.1.0, `torch` 2.14.0, `numpy` 2.5.3, `pandas` 3.0.6, `scikit-learn` 1.9.1, `httpx` 0.28.1, `huggingface_hub` 1.32.0, `python-dotenv` 1.2.3 (aggiunta nel Passo 2; lista completa in `requirements.txt`)
- Problemi / cose da sapere:
  - Nessun errore. Warning su richieste non autenticate a Hugging Face Hub: innocuo (serve `HF_TOKEN` solo per limiti più alti).
  - In `sentence-transformers` 6.x il primo parametro di `encode` si chiama `inputs` (prima `sentences`): lo passiamo in modo posizionale.
  - TMDB: la chiave v3 (32 caratteri) si passa con il parametro `api_key`; le pagine di reference parlano di Bearer token, che è l'"API Read Access Token", un'altra credenziale. Gli errori HTTP dello script non stampano l'URL, perché contiene la chiave.
  - La chiave TMDB è stata incollata in chat: rischio basso (gratuita), ma si può rigenerare dalle impostazioni API di TMDB.
  - `genre_label` in `tracks.csv` è **rumorosa** (es. Hans Zimmer "Time" = `german`): non usarla come verità nelle descrizioni, meglio le audio features.
  - Alcuni titoli dei brani hanno suffissi (`- Remastered 2011`, `- Radio Edit`, `(feat. ...)`); TMDB in inglese chiama *La grande bellezza* "The Great Beauty". Entrambi lasciati così.
  - TMDB chiede logo + avviso di attribuzione: nel README c'è l'avviso testuale, il logo va aggiunto quando ci sarà una UI.
- Decisioni prese: modello di embedding locale `all-MiniLM-L6-v2`; descrizioni in inglese, generate in un passo separato; README in inglese (repo da portfolio); CSV come formato; Spotify API scartata (audio features non disponibili per app nuove dal 27/11/2024, Premium obbligatorio dal 02/2026); TMDB per film/serie, dataset Hugging Face statico per i brani; i 20 titoli proposti da Claude e confermati da Adam, i 50 brani scelti da Claude per varietà di mood; controllo dell'overlap utenti su Amazon Reviews'23 prima del Livello 2 (soglia indicativa: almeno alcune migliaia di utenti con ≥3-5 voti in ciascun dominio); `.gitkeep` in `data/` e `notebooks/`.

## Prossimo passo (scritto dalla chat)

Da chat → Claude Code. Adam incolla qui il blocco che ricevi dalla chat.

- Passo 2 (dataset minimo) completato: vedi "Stato attuale" e Log.
- Passo 3 (proposta di Claude Code, da confermare in chat): descrizioni in inglese per i 20 film/serie e i 50 brani, da salvare accanto ai dati (es. colonna `description`). Domande aperte: chi le scrive (Claude in chat/Claude Code, con le audio features come base per i brani), lunghezza e stile (1-2 frasi su mood, ritmo, atmosfera), controllo manuale prima degli embeddings.
- Dopo: embeddings + raccomandazioni per coseno (media degli embedding dei film graditi → brani più vicini).

**Dove riprendere (sessione chiusa il 2026-09-21):** il Passo 3 non è ancora iniziato. Serve la risposta di Adam alle domande aperte qui sopra (chi scrive le descrizioni, stile, controllo manuale). Nessuna modifica pendente nel repo, ultimo commit `7f0feb1`.

**In sospeso, non bloccanti:**
- Valutare se rigenerare la chiave TMDB (è stata incollata in chat).
- Pulire i suffissi nei titoli dei brani (`- Remastered`, `- Radio Edit`, `(feat. ...)`), se serve.
- Aggiungere il logo TMDB al README quando ci sarà una UI.
- Prima del Livello 2: controllo di fattibilità su Amazon Reviews'23 (quanti utenti hanno recensito sia `Movies_and_TV` sia `CDs_and_Vinyl`; verificare licenza d'uso).

## Log

Una riga per passo completato: data, cosa, commit.

- 2026-09-21 — Passo 1: ambiente, struttura, `.gitignore`, `requirements.txt`, smoke test `(3, 384)` — commit `56d5f64` (setup), `45b0a35` (smoke test); README e CLAUDE.md nel commit `bb621cd` (docs).
- 2026-09-21 — Passo 2: dataset minimo, 20 film/serie (TMDB) e 50 brani (Hugging Face) in CSV, `.env.example` — commit `a567604` (titoli), `2b676d6` (brani), `c3e33d1` (`.env.example`); README e CLAUDE.md nel commit `7f0feb1` (docs).
