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

## Modes

It degrades instead of dying. In order:

| Mode | What it means |
|---|---|
| **worker** | Whisper on its own thread. Nothing on screen stutters. |
| **main thread** | No `GM_addElement`, so the model runs on the page thread. The page freezes for about a second per `[ __ ]`. |
| **guess only** | No model, no audio, or you turned it off. Grammar and timing guess at the word. Often right. Not always right. |

The pill in the corner says which one you are getting. Hover it for the reason.

## Firefox

Used daily, on Firefox, by a human. That is more testing than the rest of this
file can claim. Everything else was verified by a scripted Chromium driving a real
Tampermonkey — thorough about the things it thought to measure, and blind to the
things it did not. A human just watches the video.

## Notes

How it works, what was measured, and everything that broke on the way:
[`docs/NOTES.md`](docs/NOTES.md).
