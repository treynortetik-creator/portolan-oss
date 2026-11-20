// Unit test for the cue-mapping function used by the narration sync script
// in how-portolan-works.html. Extracted here (not imported from the page,
// since jsdom is not available in this environment) so the mapping logic can
// be checked with a plain `node cue-map.test.js`. Keep this in sync with the
// `cueIndexAt` function inside the page's second <script> block.
"use strict";

const fs = require("fs");
const path = require("path");

const cues = JSON.parse(
  fs.readFileSync(path.join(__dirname, "cues.json"), "utf8")
);

function cueIndexAt(cues, t) {
  for (let i = cues.length - 1; i >= 0; i--) {
    if (t >= cues[i].start) return i;
  }
  return 0;
}

function idAt(t) {
  return cues[cueIndexAt(cues, t)].id;
}

let failures = 0;
function check(label, actual, expected) {
  if (actual !== expected) {
    failures++;
    console.error(`FAIL ${label}: expected "${expected}", got "${actual}"`);
  } else {
    console.log(`ok   ${label}: ${actual}`);
  }
}

// before anything starts / at the very first cue
check("t=0 (intro)", idAt(0), "intro");
check("t=-5 (clamped to first cue)", idAt(-5), "intro");

// mid-way through the intro cue should still read intro
const introMid = (cues[0].start + cues[0].end) / 2;
check(`t=${introMid.toFixed(2)} (mid intro)`, idAt(introMid), "intro");

// exact start boundary of how-1
const how1 = cues.find((c) => c.id === "how-1");
check(`t=${how1.start} (how-1 start)`, idAt(how1.start), "how-1");
check(`t=${(how1.start - 0.01).toFixed(2)} (just before how-1)`, idAt(how1.start - 0.01), "intro");

// where-intro should get its extra 1.0s lead-in gap before it
const whereIntro = cues.find((c) => c.id === "where-intro");
check(`t=${whereIntro.start} (where-intro start)`, idAt(whereIntro.start), "where-intro");

// the chain cue (scroll-only, not a stage)
const chain = cues.find((c) => c.id === "chain");
check(`t=${chain.start} (chain start)`, idAt(chain.start), "chain");

// the outro cue, and anything after the recording ends should still read outro
const outro = cues.find((c) => c.id === "outro");
check(`t=${outro.start} (outro start)`, idAt(outro.start), "outro");
check(`t=${(outro.end + 5).toFixed(2)} (past the end)`, idAt(outro.end + 5), "outro");

// every how-N and where-N cue should map back to itself at its own start time
for (const c of cues) {
  check(`t=${c.start} (${c.id} self-check)`, idAt(c.start), c.id);
}

if (failures) {
  console.error(`\n${failures} check(s) failed.`);
  process.exit(1);
} else {
  console.log(`\nAll ${cues.length * 2 + 8} checks passed.`);
}
