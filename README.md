# YouTube Auto-Sub Uncensor

YouTube replaces the [ __ ] in auto-generated captions with `[ __ ]`.

This puts it back.

Not by asking YouTube — YouTube doesn't have it. By listening to the video and
working it out locally, with a small Whisper model on a worker thread, early
enough that the answer is already there by the time the caption shows up.

Until then you get a guess. The guess is a guess.

## Install

1. Tampermonkey or Violentmonkey.
2. Open [`yt-auto-sub-uncensor.user.js`](yt-auto-sub-uncensor.user.js).
3. Install.

First run pulls ~40 MB of model. It's the smallest one that exists.

The worker needs `GM_addElement`. Without it the model runs on the page thread and
your video freezes for about a second every time somebody says `[ __ ]`. In a
Michael Reeves video that is a lot of freezing.

## Not magic

- The model is tiny. It hears `[ __ ]` and writes `[ __ ]`. Usually.
- Non-English videos pull a second model. Another 40 MB. Sorry.
- `[ __ ]` whose audio never reached the player's buffer stays `[ __ ]`.
- Firefox is untested. It is probably fine. Probably.

## Notes

How it works, what was measured, and everything that broke on the way:
[`docs/NOTES.md`](docs/NOTES.md).
