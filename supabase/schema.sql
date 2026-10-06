-- Votes collected by the web app. Run this once in the Supabase SQL editor.
--
-- The table is append-only: every vote, change of mind or undo is a new row, and
-- src/model.py keeps the latest one per (device, titles, track). Visitors can only
-- INSERT: they cannot read, change or delete anything, so nobody can see other
-- people's votes through the public key that ships with the app.

create table if not exists public.votes (
  id          bigint generated always as identity primary key,
  device_id   text        not null check (char_length(device_id) between 8 and 64),  -- random id kept in the browser, not a person
  title_key   text        not null check (char_length(title_key) between 1 and 200), -- sorted TMDB ids joined by commas
  title_ids   text[]      not null,
  track_id    text        not null check (char_length(track_id) between 10 and 40),
  vote        text        not null check (vote in ('up', 'partly', 'nearly', 'down', 'unknown', 'undone')),
  rank        int,
  wildcard    boolean,
  personalised boolean,
  model       text        check (char_length(model) <= 20),
  client_at   timestamptz,
  created_at  timestamptz not null default now()
);

create index if not exists votes_title_track_idx on public.votes (title_key, track_id);

alter table public.votes enable row level security;

-- Only the insert permission for the public (anon) role, and nothing else.
revoke all on table public.votes from anon, authenticated;
grant insert on table public.votes to anon;

drop policy if exists "Anyone can add votes" on public.votes;
create policy "Anyone can add votes"
  on public.votes
  for insert
  to anon
  with check (true);
