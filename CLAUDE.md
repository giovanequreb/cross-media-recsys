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
├── data/            # titles.csv (film/serie), tracks.csv (brani), descriptions.csv (descrizioni in 3 parti), soundtrack_pairs.csv (coppie film/brano), embeddings.npz (vettori), model.json (parametri del modello), raw/ (download grezzi, ignorata da git)
├── src/             # smoke_test.py, fetch_titles.py, build_tracks.py, embed.py, model.py, recommend.py, evaluate.py, export_app.py, check_overlap.py
├── app/             # index.html (web app statica), data.js (generato), tmdb.svg
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
- Passo corrente: 5 — mini modello di raccomandazione (**completato**). Livello 1 completo: Passi 1-4 chiusi, vedi Log.
- Fatto (Passo 2):
  - [x] `src/fetch_titles.py`: 20 film/serie da TMDB → `data/titles.csv` (colonne: `tmdb_id`, `title`, `year`, `type` = `movie`/`tv`, `genres` separati da `|`, `overview` in inglese)
  - [x] `src/build_tracks.py`: 50 brani scelti a mano dal dataset Hugging Face `maharshipandya/spotify-tracks-dataset` → `data/tracks.csv` (colonne: `track_id`, `title`, `artist`, `album`, `genre_label`, `popularity`, `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`)
  - [x] `.env` con la chiave TMDB (ignorato da git, permessi 600) + `.env.example` committato
  - [x] README aggiornato: sezione Dati, credit TMDB, struttura, roadmap
- Fatto (Passi 3-5, stato finale):
  - [x] Dataset allargato: `data/titles.csv` 113 film/serie (i primi 57 invariati; poi 54 titoli popolari e, su richiesta di Adam per un test, *Finding Nemo* e *The Fast and the Furious*), `data/tracks.csv` 200 brani (i primi 93 invariati; gli ultimi 107 aggiunti per coprire più generi: classica, rock classico, new wave, alternative, hip-hop, pop, elettronica, soul, folk). I 37 titoli e i 43 brani nuovi sono coppie famose film/canzone (es. *Trainspotting* / "Lust For Life"). Le prime 20 e 50 righe sono rimaste identiche.
  - [x] `data/descriptions.csv`: 313 righe, colonne `item_type`, `item_id`, `name`, `emotions`, `plot`, `setting`, `sound`, `references`. `sound` (aggiunta il 2026-10-05) = genere, strumenti e ritmo; per un film, lo stile della colonna sonora. `setting` (**aggiunta su richiesta di Adam**) = ambientazione: dove e quando si svolge il film; per un brano, il luogo e l'epoca che la musica evoca. Stile **voluto da Adam**: `emotions` = tre aggettivi; `plot` = trama leggera (film) o come suona e di cosa parla (brano); `references` = riferimenti pop (epoca, scena, estetica, dove lo si è sentito). Mai il titolo o l'artista dell'elemento stesso. Tutte scritte da Claude; la versione iniziale (solo mood, una frase) è nella cronologia git, commit `e0f0677`.
  - [x] `data/soundtrack_pairs.csv`: 83 coppie (`tmdb_id`, `track_id`, `title`, `track`), 45 titoli coinvolti.
  - [x] `src/embed.py`: un embedding per ogni parte → `data/embeddings.npz` (`item_type`, `item_id`, `facets`, `vectors` di forma 313×5×384, circa 2.4 MB, committato).
  - [x] `src/model.py` + `data/model.json`: punteggio = somma pesata delle similarità per parte − correzione "hub" (media del punteggio del brano su tutti i titoli). Pesi: emozioni 0.26, trama 0.04, ambientazione 0.19, suono 0.25, riferimenti 0.26, tone 0.0; `hub_correction` 1.0. `tone` è una sesta parte non testuale: confronta energia e positività del film (`data/title_tone.csv`, valori assegnati a mano da Claude) con `energy` e `valence` del brano; è calcolata in `model.py`, non negli embeddings.
  - [x] `src/recommend.py`: `python src/recommend.py "Drive" "Amélie" --top 5`; con più titoli fa la media dei punteggi.
  - [x] `src/evaluate.py`: controllo sulle 57 coppie, confronto tra configurazioni e ricerca a griglia dei pesi con cross-validation a 5 fold per titolo.
  - [x] **Risultati attuali (2026-10-05): 113 titoli, 200 brani, 83 coppie, 5 parti + tone** (caso = 0.03): solo tone 0.04; solo trama 0.06; solo emozioni 0.25; solo ambientazione 0.33; solo suono 0.43; solo riferimenti 0.70; **modello 0.68** (hit@1 0.63, hit@5 0.75, hit@10 0.80, rank medio 8.3). La ricerca a griglia (passi da 0.2) sceglierebbe 60% riferimenti, 20% suono, 20% ambientazione (MRR 0.75, 0.72 sui titoli tenuti fuori).
  - [x] **Due idee provate e scartate dal modello di default**: (1) `tone` con i dati audio di Spotify: da solo vale quanto il caso (0.04) e sopra il peso 0.05 abbassa il punteggio; resta come slider a 0 nell'app. Non corregge nemmeno i casi per cui era pensato (*Dune* → "Circle of Life", *Whiplash* → "Reptilia" restano primi). (2) Aggiungere "Directed by X" al testo dei riferimenti: MRR da 0.61 a 0.56. Il regista (colonna `director` in `titles.csv`, per le serie i creatori) è quindi salvato e mostrato nell'app ma **non** entra negli embeddings.
  - [x] Risultati precedenti con 4 parti: 111 titoli, 200 brani, 83 coppie, modello `BAAI/bge-small-en-v1.5` (caso = 0.03): solo trama 0.06; solo emozioni 0.25; solo ambientazione 0.33; solo riferimenti 0.70; pesi uguali 0.53; **modello 0.61** (hit@1 0.53, hit@5 0.67, hit@10 0.71, rank medio 9.9 su 200). Varietà: nei top-5 dei 111 titoli compaiono 171 brani diversi, il più ripetuto 12 volte (senza correzione hub: 157 e 22). La ricerca a griglia sceglierebbe 70% riferimenti (MRR 0.74, 0.72 sui titoli tenuti fuori).
  - [x] **Confronto tra modelli di embedding** sulle 83 coppie, stesse descrizioni e pesi (MRR): `all-MiniLM-L6-v2` 0.52; `all-MiniLM-L12-v2` 0.53; `all-mpnet-base-v2` 0.52; `gte-base` 0.59; `mxbai-embed-large-v1` 0.59; `bge-base-en-v1.5` 0.60; **`bge-small-en-v1.5` 0.61** (scelto: 130 MB, sempre 384 dimensioni); `bge-large-en-v1.5` 0.63 (scartato: 1.3 GB e 8 volte più lento per +0.02). I modelli scartati sono stati cancellati dalla cache di Hugging Face.
  - [x] Risultati precedenti con 57 titoli, 57 coppie e `all-MiniLM-L6-v2` (caso = 0.03): solo trama 0.05; solo emozioni 0.26; solo ambientazione 0.45; solo riferimenti 0.71; **modello 0.61** (hit@1 0.51, hit@5 0.74, hit@10 0.82, rank medio 11.0 su 200). Il numero è più basso di prima perché il compito è più difficile (il doppio dei brani sbagliati tra cui scegliere), non perché il modello sia peggiorato. Varietà: nei top-5 dei 57 titoli compaiono 153 brani diversi, il più ripetuto 7 volte (senza correzione hub: 123 e 10). La ricerca a griglia sceglierebbe 80% riferimenti (MRR 0.77, 0.73 sui titoli tenuti fuori).
  - [x] Risultati precedenti con 93 brani (MRR, caso = 0.06): solo trama 0.07; solo emozioni 0.31; solo ambientazione 0.49; solo riferimenti 0.74; pesi uguali 0.66; **modello 0.67** (hit@1 0.58, hit@5 0.81, hit@10 0.86, rank medio 5.6). Prima dell'ambientazione il modello era a 0.65 con hit@5 0.75. Correzione hub: il brano più ripetuto nei top-5 passa da 17 titoli su 57 a 9.
- Fatto (app web):
  - [x] `app/index.html`: pagina statica senza backend e senza dipendenze (HTML + CSS + JS in un file). Si scelgono i titoli, uno slider per ogni parte della descrizione cambia i pesi in tempo reale, una casella attiva la correzione hub, ogni brano ha il link a Spotify e i pulsanti 👍/👎. La formula in JS è la stessa di `src/model.py`.
  - [x] Pulsante **Listen** su ogni brano: apre il player di Spotify (anteprima) dentro la pagina; votare non ricarica la lista, quindi il brano continua a suonare. Sotto ogni brano si leggono emozioni, suono e riferimenti. Il regista compare accanto al titolo selezionato e nel tooltip di ogni film.
  - [x] **Rating mode** (casella sopra i risultati): mostra gli 8 migliori più 3 brani "a sorpresa" pescati tra le posizioni 9-60, in ordine mescolato e senza posizione né barra del punteggio. L'ordine è deterministico per ogni selezione, quindi non cambia dopo un voto. Ogni voto salva anche `rank` (posizione vera) e `wildcard`. Scala a quattro livelli **chiesta da Adam** (ascolta i brani e trova casi intermedi), senza via di mezzo neutra: `up` (++, ci sta), `partly` (+, ci sta un pochino), `nearly` (−, non ci sta ma non è troppo fuori), `down` (−−, non ci sta), più `unknown` (?, non conosco il brano) da escludere quando si imparano i pesi. I voti dati prima di questa modifica sono solo `up`/`down`.
  - [x] Piano concordato con Adam per i voti: un film alla volta, tutti gli 11 brani, circa 20-25 film diversi tra loro, slider sui valori di default; poi scaricare `feedback.json` e metterlo nella cartella del progetto per imparare i pesi.
  - [x] I voti restano nel `localStorage` del browser e si scaricano come `feedback.json` (titoli scelti, brano, voto, pesi in uso, data): sono i dati per imparare i pesi.
  - [x] `src/export_app.py` genera `app/data.js` (titoli, brani, descrizioni, similarità per parte; circa 430 KB). **Va rieseguito** dopo ogni modifica a descrizioni, embeddings o `data/model.json`.
  - [x] Avvio: `python -m http.server 5173 --directory app` (configurato anche in `.claude/launch.json` come server `app`), oppure doppio clic su `app/index.html`.
  - [x] Logo e avviso TMDB nel footer dell'app.
  - [x] **App pubblicata** (2026-10-05, con l'ok di Adam): https://giovanequreb.github.io/cross-media-recsys/ . GitHub Pages è in modalità "GitHub Actions"; il workflow `.github/workflows/pages.yml` ripubblica la cartella `app/` a ogni push su `main` che tocca `app/` (circa 30 secondi). Quindi dopo `src/export_app.py` basta committare e fare push perché il sito si aggiorni. I voti restano nel browser di chi vota: chi prova l'app deve scaricare `feedback.json` e mandarlo ad Adam.
- Fatto (verso il Livello 2): `src/check_overlap.py` legge in streaming (niente su disco) i file rating-only di Amazon Reviews'23, `CDs_and_Vinyl` (4.772.071 voti, 1.754.118 utenti) e `Movies_and_TV` (17.158.519 voti). Utenti con almeno N voti in **entrambi** i domini: N≥1 713.375; N≥3 123.527; N≥5 54.260; N≥10 17.292; N≥20 5.153. **Il Livello 2 è fattibile**: la soglia indicativa (alcune migliaia di utenti con ≥3-5 voti per dominio) è superata di molto. Dura circa 2 minuti.
- Versioni installate: Python 3.12.3, `sentence-transformers` 6.1.0, `torch` 2.14.0, `numpy` 2.5.3, `pandas` 3.0.6, `scikit-learn` 1.9.1, `httpx` 0.28.1, `huggingface_hub` 1.32.0, `python-dotenv` 1.2.3 (aggiunta nel Passo 2; lista completa in `requirements.txt`)
- Problemi / cose da sapere:
  - Nessun errore. Warning su richieste non autenticate a Hugging Face Hub: innocuo (serve `HF_TOKEN` solo per limiti più alti).
  - In `sentence-transformers` 6.x il primo parametro di `encode` si chiama `inputs` (prima `sentences`): lo passiamo in modo posizionale.
  - TMDB: la chiave v3 (32 caratteri) si passa con il parametro `api_key`; le pagine di reference parlano di Bearer token, che è l'"API Read Access Token", un'altra credenziale. Gli errori HTTP dello script non stampano l'URL, perché contiene la chiave.
  - La chiave TMDB è stata incollata in chat: rischio basso (gratuita), ma si può rigenerare dalle impostazioni API di TMDB.
  - `genre_label` in `tracks.csv` è **rumorosa** (es. Hans Zimmer "Time" = `german`): non usarla come verità nelle descrizioni, meglio le audio features.
  - Alcuni titoli dei brani hanno suffissi (`- Remastered 2011`, `- Radio Edit`, `(feat. ...)`); TMDB in inglese chiama *La grande bellezza* "The Great Beauty". Entrambi lasciati così.
  - **I pesi non sono imparati dai dati, e per un motivo preciso**: la ricerca a griglia sceglierebbe 90% riferimenti e 10% ambientazione (MRR 0.80, 0.79 sui titoli tenuti fuori), perché i riferimenti di un brano spesso descrivono proprio la scena in cui è famoso ("boxing training montage"). Quel controllo misura "ritrovo le canzoni famose di un film", non "questo brano piace a chi ama quel film". Adam vuole emozioni + riferimenti pop, quindi i pesi sono fissati a mano; per impararli davvero servono feedback reali (Livello 3).
  - Il controllo sulle colonne sonore è **ottimistico**: le descrizioni le ha scritte un LLM che conosce le opere e le coppie.
  - `src/recommend.py` e `src/evaluate.py` importano da `model.py` (`from model import ...`): funziona perché si lancia come `python src/evaluate.py`, che mette `src/` nel percorso degli import. Dopo ogni modifica a `data/descriptions.csv` va rieseguito `src/embed.py`.
  - Amazon Reviews'23: **nessuna licenza esplicita** né sul sito né sulla scheda Hugging Face (`McAuley-Lab/Amazon-Reviews-2023`); gli autori chiedono solo la citazione (Hou et al., 2024, arXiv 2403.03952). Trattarlo come uso di ricerca, non committare dati grezzi. I file usano `parent_asin` come id del prodotto: per mostrare titoli veri servono anche i file di metadati (molto più grandi).
  - Il disco del Mac è quasi pieno (circa 3 GB liberi il 2026-10-04): per il Livello 2 non scaricare i file interi senza prima liberare spazio; lo streaming funziona.
  - TMDB chiede logo + avviso di attribuzione: nel README c'è l'avviso testuale, il logo va aggiunto quando ci sarà una UI.
- Repo GitHub (pubblico): https://github.com/giovanequreb/cross-media-recsys, remote `origin`, branch `main`. I commit di questo repo sono firmati con l'email personale (config git locale); il 2026-10-04 la cronologia è stata riscritta per sostituire l'email, quindi gli hash nel Log sono quelli nuovi.
- Decisioni prese: modello di embedding locale `BAAI/bge-small-en-v1.5` (all'inizio `all-MiniLM-L6-v2`); descrizioni in inglese, generate in un passo separato; README in inglese (repo da portfolio); CSV come formato; Spotify API scartata (audio features non disponibili per app nuove dal 27/11/2024, Premium obbligatorio dal 02/2026); TMDB per film/serie, dataset Hugging Face statico per i brani; i 20 titoli proposti da Claude e confermati da Adam, i 50 brani scelti da Claude per varietà di mood; controllo dell'overlap utenti su Amazon Reviews'23 prima del Livello 2 (soglia indicativa: almeno alcune migliaia di utenti con ≥3-5 voti in ciascun dominio); `.gitkeep` in `data/` e `notebooks/`.

## Prossimo passo (scritto dalla chat)

Da chat → Claude Code. Adam incolla qui il blocco che ricevi dalla chat.

- Livello 1 completato (Passi 1-5), con un mini modello a pesi. Prossimo: raccogliere feedback reali per imparare i pesi, oppure Livello 2.

### Come riprendere in una nuova conversazione

Aggiornato il 2026-10-04. Il Livello 1 è **completo e pubblicato su GitHub**: dataset (57 titoli, 93 brani), descrizioni in tre parti, embeddings, modello a pesi con correzione hub, raccomandazioni da riga di comando e controllo su 57 coppie film/brano (controlla con `git status` e `git log --oneline`).

**All'avvio, in ordine:**
1. Leggi questo file (viene caricato da solo) e rispetta le "Regole di lavoro": un passo alla volta, codice Python spiegato in italiano con paragoni JS/TS, verifica documentazione e versioni prima di usare una libreria, commit piccoli, niente segreti nel repo.
2. Non rifare quello che è già fatto (vedi "Stato attuale" e Log).
3. Ambiente: usa sempre `.venv/bin/python` (o `source .venv/bin/activate`). La chiave TMDB è già nel `.env` locale (non nel repo): non chiederla di nuovo e non stamparla. Serve solo per rieseguire `src/fetch_titles.py`.
4. Il 2026-10-04 Adam ha chiesto di procedere senza chiedere conferme in quella sessione. In una nuova conversazione vale di nuovo la regola "se una scelta è ambigua, chiedi".

**Prossimi passi possibili (da decidere):**
- Far provare i consigli ad Adam e a 5-10 persone, come previsto prima del Livello 2.
- Livello 2: scegliere il modello (es. fattorizzazione di matrice / two-tower sugli utenti con ≥5 voti per dominio) e come collegare i prodotti Amazon a titoli e brani veri (servono i metadati).
- Dopo l'aggiunta dell'ambientazione Adam ha provato l'app e ha detto che i consigli funzionano bene (2026-10-04). Limiti noti che restano: catalogo ancora limitato (200 brani), modello di embedding piccolo, pesi fissati a mano.
- Adam vuole che il progetto diventi un **mini modello di raccomandazione**: il passo naturale è raccogliere giudizi veri con l'app ("questo brano ci sta / non ci sta" per un film) e imparare pesi e correzioni da quelli, invece che fissarli a mano.
- Allargare ancora il catalogo dei brani generando le descrizioni con uno script e un'API LLM (a pagamento).

**In sospeso, non bloccanti:**
- Valutare se rigenerare la chiave TMDB (è stata incollata in chat).
- Pulire i suffissi nei titoli dei brani (`- Remastered`, `- Radio Edit`, `(feat. ...)`), se serve.

## Log

Una riga per passo completato: data, cosa, commit.

- 2026-09-21 — Passo 1: ambiente, struttura, `.gitignore`, `requirements.txt`, smoke test `(3, 384)` — commit `6209479` (setup), `e088a3d` (smoke test); README e CLAUDE.md nel commit `97c14c5` (docs).
- 2026-09-21 — Passo 2: dataset minimo, 20 film/serie (TMDB) e 50 brani (Hugging Face) in CSV, `.env.example` — commit `0498385` (titoli), `74b47b2` (brani), `af412f4` (`.env.example`); README e CLAUDE.md nel commit `f4cefc4` (docs).
- 2026-10-04 — Passo 3: 70 descrizioni di mood in `data/descriptions.csv` — commit `37dd7b4`, correzioni in `e0f0677`. Repo pubblicato su GitHub lo stesso giorno.
- 2026-10-04 — Passo 4: embeddings e raccomandazioni per coseno, controllo sulle colonne sonore (MRR 0.68) — commit `b5d3056` (embed), `a854c9a` (recommend), `7abefb2` (evaluate). Livello 1 completo.
- 2026-10-04 — Fattibilità Livello 2: 54.260 utenti con ≥5 voti sia in film sia in musica su Amazon Reviews'23 — commit `0e524da`.
- 2026-10-04 — Descrizioni riscritte su richiesta di Adam (emozioni, trama leggera, riferimenti pop), embeddings rigenerati, MRR 0.63 — commit `a8cff33`.
- 2026-10-04 — Passo 5: dataset a 57 titoli e 93 brani, descrizioni in tre parti, modello a pesi con correzione hub, controllo su 57 coppie (MRR 0.65) — commit `34e9bbc` (modello); dataset nel commit precedente.
- 2026-10-04 — App web statica (`app/`) con pesi regolabili e raccolta dei voti; `src/export_app.py`.
- 2026-10-04 — Quarta parte `setting` (ambientazione) su richiesta di Adam: modello a 4 pesi, MRR 0.67, hit@5 0.81.
- 2026-10-04 — Catalogo allargato a 200 brani (107 nuovi con descrizioni in quattro parti); MRR 0.61 su 200, consigli più vari.
- 2026-10-04 — 54 titoli nuovi (111 in tutto), 83 coppie, modello di embedding cambiato in `BAAI/bge-small-en-v1.5` dopo un confronto tra 8 modelli; MRR 0.61.
- 2026-10-04 — Rating mode nell'app (lista mescolata con 3 brani a sorpresa) per raccogliere i voti di Adam.
- 2026-10-05 — Parte `sound` (MRR da 0.61 a 0.68), `tone` sperimentale a peso 0, registi da TMDB, player Spotify nell'app.
- 2026-10-05 — App pubblicata su GitHub Pages con deploy automatico — commit `390f202`.
