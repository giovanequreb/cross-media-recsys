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
├── data/            # titles.csv (film/serie), tracks.csv (brani), descriptions.csv (descrizioni di mood), embeddings.npz (vettori), raw/ (download grezzi, ignorata da git)
├── src/             # smoke_test.py, fetch_titles.py, build_tracks.py, embed.py, recommend.py, evaluate.py
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
- Passo corrente: 4 — embeddings e raccomandazioni (**completato**): il Livello 1 funziona da capo a fondo. I Passi 2 (dataset minimo) e 3 (descrizioni) sono chiusi. Il Passo 1 (ambiente, struttura, smoke test `(3, 384)`) è chiuso: vedi Log.
- Fatto (Passo 2):
  - [x] `src/fetch_titles.py`: 20 film/serie da TMDB → `data/titles.csv` (colonne: `tmdb_id`, `title`, `year`, `type` = `movie`/`tv`, `genres` separati da `|`, `overview` in inglese)
  - [x] `src/build_tracks.py`: 50 brani scelti a mano dal dataset Hugging Face `maharshipandya/spotify-tracks-dataset` → `data/tracks.csv` (colonne: `track_id`, `title`, `artist`, `album`, `genre_label`, `popularity`, `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`)
  - [x] `.env` con la chiave TMDB (ignorato da git, permessi 600) + `.env.example` committato
  - [x] README aggiornato: sezione Dati, credit TMDB, struttura, roadmap
- Fatto (Passo 3):
  - [x] `data/descriptions.csv`: 70 righe (20 film/serie + 50 brani), colonne `item_type` (`title`/`track`), `item_id` (`tmdb_id` o `track_id`), `name` (solo per leggibilità, **non va negli embeddings**), `description`
  - [x] Stile: tre aggettivi di mood + una frase su ritmo e atmosfera, in inglese, senza nomi propri; 25-51 token (limite del modello: 256)
  - [x] Campione di 10 scritto da Claude Code e approvato da Adam; le altre 60 scritte da tre subagent con lo stesso schema, poi unite e controllate (70 descrizioni uniche, nessuna parola del titolo/artista nel testo)
  - [x] Revisione fatta da Claude Code al posto di Adam (su sua richiesta di procedere senza conferme): corrette 5 descrizioni che citavano il testo delle canzoni o parole legate al cinema. Adam può comunque rileggerle: dopo ogni modifica va rieseguito `src/embed.py`
- Fatto (Passo 4):
  - [x] `src/embed.py`: legge `data/descriptions.csv`, codifica solo la colonna `description` con `normalize_embeddings=True` e salva `data/embeddings.npz` (array `item_type`, `item_id`, `vectors` 70×384 float32, committato: 116 KB)
  - [x] `src/recommend.py`: `python src/recommend.py "Drive" "Amélie" --top 5` → media dei vettori dei titoli graditi, rinormalizzata, poi prodotto scalare con i 50 brani. I titoli si cercano senza distinguere maiuscole, anche per sottostringa (`"mad max"`)
  - [x] `src/evaluate.py`: controllo sulle 5 coppie titolo/brano della propria colonna sonora presenti nel dataset. Risultato: rank 1, 1, 3, 1, 24 su 50; MRR 0.68 contro 0.09 del caso
  - [x] Varietà: sui 20 titoli, nei top-5 compaiono 36 brani diversi su 50 (nessun brano "pigliatutto")
- Versioni installate: Python 3.12.3, `sentence-transformers` 6.1.0, `torch` 2.14.0, `numpy` 2.5.3, `pandas` 3.0.6, `scikit-learn` 1.9.1, `httpx` 0.28.1, `huggingface_hub` 1.32.0, `python-dotenv` 1.2.3 (aggiunta nel Passo 2; lista completa in `requirements.txt`)
- Problemi / cose da sapere:
  - Nessun errore. Warning su richieste non autenticate a Hugging Face Hub: innocuo (serve `HF_TOKEN` solo per limiti più alti).
  - In `sentence-transformers` 6.x il primo parametro di `encode` si chiama `inputs` (prima `sentences`): lo passiamo in modo posizionale.
  - TMDB: la chiave v3 (32 caratteri) si passa con il parametro `api_key`; le pagine di reference parlano di Bearer token, che è l'"API Read Access Token", un'altra credenziale. Gli errori HTTP dello script non stampano l'URL, perché contiene la chiave.
  - La chiave TMDB è stata incollata in chat: rischio basso (gratuita), ma si può rigenerare dalle impostazioni API di TMDB.
  - `genre_label` in `tracks.csv` è **rumorosa** (es. Hans Zimmer "Time" = `german`): non usarla come verità nelle descrizioni, meglio le audio features.
  - Alcuni titoli dei brani hanno suffissi (`- Remastered 2011`, `- Radio Edit`, `(feat. ...)`); TMDB in inglese chiama *La grande bellezza* "The Great Beauty". Entrambi lasciati così.
  - Il controllo sulle colonne sonore è **ottimistico**: solo 5 coppie, e le descrizioni le ha scritte un LLM che conosce le opere, quindi parte della corrispondenza può venire da come sono scritte. Il caso debole è *Pulp Fiction* → "Son Of A Preacher Man" (rank 24).
  - `src/evaluate.py` importa da `recommend.py` (`from recommend import ...`): funziona perché si lancia come `python src/evaluate.py`, che mette `src/` nel percorso degli import.
  - TMDB chiede logo + avviso di attribuzione: nel README c'è l'avviso testuale, il logo va aggiunto quando ci sarà una UI.
- Repo GitHub (pubblico): https://github.com/giovanequreb/cross-media-recsys, remote `origin`, branch `main`. I commit di questo repo sono firmati con l'email personale (config git locale); il 2026-10-04 la cronologia è stata riscritta per sostituire l'email, quindi gli hash nel Log sono quelli nuovi.
- Decisioni prese: modello di embedding locale `all-MiniLM-L6-v2`; descrizioni in inglese, generate in un passo separato; README in inglese (repo da portfolio); CSV come formato; Spotify API scartata (audio features non disponibili per app nuove dal 27/11/2024, Premium obbligatorio dal 02/2026); TMDB per film/serie, dataset Hugging Face statico per i brani; i 20 titoli proposti da Claude e confermati da Adam, i 50 brani scelti da Claude per varietà di mood; controllo dell'overlap utenti su Amazon Reviews'23 prima del Livello 2 (soglia indicativa: almeno alcune migliaia di utenti con ≥3-5 voti in ciascun dominio); `.gitkeep` in `data/` e `notebooks/`.

## Prossimo passo (scritto dalla chat)

Da chat → Claude Code. Adam incolla qui il blocco che ricevi dalla chat.

- Livello 1 completato (Passi 1-4). Prossimo: decidere come procedere verso il Livello 2.

### Come riprendere in una nuova conversazione

Aggiornato il 2026-10-04. Il Livello 1 è **completo e pubblicato su GitHub**: dataset, descrizioni, embeddings, raccomandazioni da riga di comando e controllo sulle colonne sonore (controlla con `git status` e `git log --oneline`).

**All'avvio, in ordine:**
1. Leggi questo file (viene caricato da solo) e rispetta le "Regole di lavoro": un passo alla volta, codice Python spiegato in italiano con paragoni JS/TS, verifica documentazione e versioni prima di usare una libreria, commit piccoli, niente segreti nel repo.
2. Non rifare quello che è già fatto (vedi "Stato attuale" e Log).
3. Ambiente: usa sempre `.venv/bin/python` (o `source .venv/bin/activate`). La chiave TMDB è già nel `.env` locale (non nel repo): non chiederla di nuovo e non stamparla. Serve solo per rieseguire `src/fetch_titles.py`.
4. Il 2026-10-04 Adam ha chiesto di procedere senza chiedere conferme in quella sessione. In una nuova conversazione vale di nuovo la regola "se una scelta è ambigua, chiedi".

**Prossimi passi possibili (da decidere):**
- Far provare i consigli ad Adam e a 5-10 persone, come previsto prima del Livello 2.
- Controllo di fattibilità del Livello 2 su Amazon Reviews'23 (vedi "In sospeso").
- Allargare il dataset del Livello 1 (più titoli e brani) generando le descrizioni con uno script.

**In sospeso, non bloccanti:**
- Valutare se rigenerare la chiave TMDB (è stata incollata in chat).
- Pulire i suffissi nei titoli dei brani (`- Remastered`, `- Radio Edit`, `(feat. ...)`), se serve.
- Aggiungere il logo TMDB al README quando ci sarà una UI.
- Prima del Livello 2: controllo di fattibilità su Amazon Reviews'23 (quanti utenti hanno recensito sia `Movies_and_TV` sia `CDs_and_Vinyl`; verificare licenza d'uso).

## Log

Una riga per passo completato: data, cosa, commit.

- 2026-09-21 — Passo 1: ambiente, struttura, `.gitignore`, `requirements.txt`, smoke test `(3, 384)` — commit `6209479` (setup), `e088a3d` (smoke test); README e CLAUDE.md nel commit `97c14c5` (docs).
- 2026-09-21 — Passo 2: dataset minimo, 20 film/serie (TMDB) e 50 brani (Hugging Face) in CSV, `.env.example` — commit `0498385` (titoli), `74b47b2` (brani), `af412f4` (`.env.example`); README e CLAUDE.md nel commit `f4cefc4` (docs).
- 2026-10-04 — Passo 3: 70 descrizioni di mood in `data/descriptions.csv` — commit `37dd7b4`, correzioni in `e0f0677`. Repo pubblicato su GitHub lo stesso giorno.
- 2026-10-04 — Passo 4: embeddings e raccomandazioni per coseno, controllo sulle colonne sonore (MRR 0.68) — commit `b5d3056` (embed), `a854c9a` (recommend), `7abefb2` (evaluate). Livello 1 completo.
