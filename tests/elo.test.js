// Run with: node tests/elo.test.js
const assert = require("node:assert/strict");
const Elo = require("../app/elo.js");

// Equal ratings: 50% chance, so a win moves each side by K/2 and the total stays constant.
assert.equal(Elo.expected(1000, 1000), 0.5);
const [w, l] = Elo.update(1000, 1000);
assert.equal(w, 1016);
assert.equal(l, 984);
assert.equal(w + l, 2000);

// An upset (the underdog wins) moves ratings more than an expected win.
const upset = Elo.update(900, 1100)[0] - 900;
const routine = Elo.update(1100, 900)[0] - 1100;
assert.ok(upset > routine);

// 400 points above counts ten times as much.
assert.ok(Math.abs(Elo.strength(1400) / Elo.strength(1000) - 10) < 1e-9);

// Pair selection: two different titles, favouring the ones with fewest duels.
const ids = ["a", "b", "c", "d"];
const games = { a: 3, b: 0, c: 3, d: 3 };
for (let i = 0; i < 50; i++) {
  const pair = Elo.nextPair(ids, {}, games, new Set());
  assert.notEqual(pair[0], pair[1]);
  assert.ok(pair.includes("b"));
}
assert.equal(Elo.nextPair(["a"], {}, {}, new Set()), null);

// Pairs already seen are avoided while other options exist.
const seen = new Set(["a|b", "a|c"]);
for (let i = 0; i < 50; i++) {
  const pair = Elo.nextPair(["a", "b", "c", "d"], {}, { a: 0, b: 5, c: 5, d: 5 }, seen);
  assert.deepEqual([...pair].sort(), ["a", "d"]);
}

// Simulated duels: the title that always wins ends up on top.
const ratings = { x: Elo.START, y: Elo.START, z: Elo.START };
for (let i = 0; i < 30; i++) {
  for (const [win, lose] of [["x", "y"], ["x", "z"], ["y", "z"]]) {
    [ratings[win], ratings[lose]] = Elo.update(ratings[win], ratings[lose]);
  }
}
assert.ok(ratings.x > ratings.y && ratings.y > ratings.z);

console.log("elo.test.js: all checks passed");
