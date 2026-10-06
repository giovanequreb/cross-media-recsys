// Elo ranking from pairwise choices (the idea behind Podiums): every title starts
// at the same rating; each duel moves the winner up and the loser down, by more
// when the result was unexpected. Pure functions, so they can be tested in Node.
const Elo = (() => {
  const START = 1000;
  const K = 32; // how far one duel can move a rating

  // Probability that a title rated `a` beats one rated `b` (logistic curve, 400-point scale).
  const expected = (a, b) => 1 / (1 + 10 ** ((b - a) / 400));

  // New ratings after a duel.
  function update(winner, loser, k = K) {
    const gain = k * (1 - expected(winner, loser));
    return [winner + gain, loser - gain];
  }

  // Elo strength 10^(rating/400): a title 400 points above another counts 10 times as much.
  const strength = (rating) => 10 ** (rating / 400);

  // Choose the next duel: a title with few duels so far against one of similar
  // rating (those duels carry the most information), avoiding pairs already seen.
  function nextPair(ids, ratings, games, seenPairs, random = Math.random) {
    if (ids.length < 2) return null;
    const rating = (id) => ratings[id] ?? START;
    const played = (id) => games[id] ?? 0;
    const fewest = Math.min(...ids.map(played));
    const pool = ids.filter((id) => played(id) === fewest);
    const first = pool[Math.floor(random() * pool.length)];
    const key = (a, b) => [a, b].sort().join("|");
    const others = ids.filter((id) => id !== first);
    const fresh = others.filter((id) => !seenPairs.has(key(first, id)));
    const candidates = fresh.length ? fresh : others;
    // a little noise, so the second title is not always the same one
    const scored = candidates.map((id) => [Math.abs(rating(id) - rating(first)) + random() * 60, id]);
    const second = scored.reduce((best, item) => (item[0] < best[0] ? item : best))[1];
    return random() < 0.5 ? [first, second] : [second, first];
  }

  return { START, K, expected, update, strength, nextPair };
})();

if (typeof module !== "undefined") module.exports = Elo;
