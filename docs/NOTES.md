# Notes

Working record: what was measured, what was built, and what broke.

> Redacted before publishing: cookie values, session tokens (`pot=`, `signature=`),
> and local file paths from the debugging session. Nothing else was changed.

---

## 1. What YouTube actually does

Measured against a live video, not inferred.

- **The censorship is server-side.** The timedtext payload contains
  `"[\u00a0__\u00a0]"` — NBSP-padded, not a display filter. There is no hidden field
  holding the original.
- **No format or client leaks it.** `fmt=json3|srv1|srv3|vtt|ttml` and
  `c=ANDROID|IOS|TVHTML5|MWEB` return byte-identical censored text.
- **`tlang=` is not an escape hatch.** It returns `429 Sorry...` from this IP.
- **The censor list is inconsistent.** In one video `shitty`, `shit's hard` and
  `smack him around` survive untouched while `shit` in `slow as shit` does not.
- **`pot` is now required.** Without a proof-of-origin token the endpoint answers
  `HTTP 200` with `content-length: 0`. The player appends
  `pot=…&potc=1&c=WEB&cver=…&cbr=…&cos=…&cplatform=DESKTOP` to its own caption URL,
  so the script captures that URL instead of building one.
- **The player sends `cbr=HeadlessChrome` when it thinks it is headless, and the
  endpoint returns empty for that.** Relevant only to testing; a real browser sends
  `cbr=Chrome`/`Firefox`.

Conclusion: the word is not in YouTube's data. It has to come from the audio.

## 2. Where the audio comes from

Recording the player's live output with `captureStream()` works, but it can only
ever produce audio the playhead has **already passed**, so a correction can never
arrive before the caption it belongs to. That was the first working version and it
was the wrong shape.

The audio that exists *ahead* of the playhead is what the player has already
buffered. Measured:

| Fact | Result |
|---|---|
| Audio `SourceBuffer` mime | `audio/webm; codecs="opus"`, EBML magic `0x1A45DFA3` |
| Cluster timecodes | absolute milliseconds (0 / 10001 / 20001 …) |
| Decoding `header + cluster group` | 9.99 s of audio in **29 ms** |
| `sb.mimeType` at append time | reads back empty; the type is remembered at `addSourceBuffer` |

So: hook `MediaSource.prototype.addSourceBuffer`, keep a copy of the audio bytes,
index the Clusters by timecode, and decode with `decodeAudioData` — the browser's
own decoder, off the JS main thread.

`decodeAudioData` also defines presentation time as `container timestamp +
sb.timestampOffset`, so that offset is recorded per append. Without it, a range the
player fetched for a seek lands on the wrong timeline.

### Ad breaks

Ads run through the same pipeline and their Clusters also start near zero. Capturing
them splices two unrelated videos onto one timeline — which is how the transcript
`"We plan the whole area around this defining feature, the water"` ended up
answering a cue from a completely different video. Appends are ignored while
`.ad-showing` is present, and the archive resynchronises on the EBML header rather
than trusting append order.

## 3. Picking the word out of the transcript

The censored word is its own json3 seg with a `tOffsetMs`, so its time is known.
That is not enough.

YouTube's per-word offsets are approximate. Measured against Whisper's word
timestamps on the same audio:

```
YouTube: "good"   at 63.57s     Whisper: 63.80s     -> 230 ms apart
YouTube: "[ __ ]" at 63.84s     Whisper: 63.94s     -> 100 ms apart
"slow as [ __ ]"                                     -> ~400 ms apart
```

400 ms is wider than the gap between words, so nearest-onset matching picks the
word next door — `slow as [__]` came back as `as`. Containment is worse: it flips at
the boundary.

What works is **alignment against the caption's own word sequence**. The words
either side of the gap are known exactly, so the censored word is whatever Whisper
puts at that position. Validated on real speech: 22/22 words recovered, and every
censored word in one cue is filled from a single transcription.

## 4. Keeping the page alive

A single Whisper call has a floor of about **1.5 s**: the mel spectrogram is always
padded to 30 s, so the encoder cost is fixed however short the clip is. Measured:

```
 2 s clip -> 1520 ms      9 s clip -> 2716 ms
 3 s clip -> 1890 ms     11 s clip -> 3022 ms
```

Shortening the clip cannot help. On the main thread that is a visible freeze per
censored word.

Things that were tried and did not work:

| Approach | Result |
|---|---|
| Blob `Worker` from the page | refused by the page CSP (no `blob:` in `script-src`) |
| Sandboxed iframe, opaque origin | **same thread** — a 2.5 s busy loop inside froze the parent solid |
| `data:` iframe via plain `createElement` | its script never runs; it still inherits the page policy |

What works is `GM_addElement`: a `data:` iframe injected privileged does not carry
the page's policy, and a `Worker` built inside it runs on a real thread. Inference
goes there, with a main-thread fallback.

Verified with Tampermonkey: `page.evaluate` round-trips stayed at **4–7 ms**
throughout a 3376 ms transcription. The same call on the main thread times out.

## 5. Bugs found while verifying

Each of these was reproduced, fixed, and re-tested. They are listed because the
symptom rarely points at the cause.

**Caption capture**

- `durMs` used outside its scope in `parseTrack` — every parse threw, no cues at all.
- Pre-roll ad tracks were accepted and overwrote the video's own captions.
- `yt-navigate-finish` fires *after* the caption track is captured, wiping it.
- The player re-fetches the track with a fresh URL on reinitialise, and re-parsing
  rebuilt every guess — silently discarding everything Whisper had answered.

**Audio**

- `getVideo()` returned "the first `<video>` in the document", which can be a paused
  sidebar preview. `pumpQueue` bailed on `video.paused` and nothing was ever
  transcribed.
- The cluster scan skipped the first byte of each append — which is exactly where a
  Cluster starts.
- The archive was reset on video change while the same `SourceBuffer` kept
  appending, leaving a header-less stream that could never decode.
- A gap in the ring buffer was invisible to a span-only check, so two unrelated
  islands of audio were spliced into a sentence nobody spoke.
- Pausing for more than three seconds wiped the buffer.

**Word selection**

- `cleanWord` rejected any word carrying Whisper's punctuation, so `" shit,"` was
  dropped as invalid.
- The two sides of the alignment used different normalizers, so an apostrophe
  stopped a word from ever matching.

**Worker**

- The handshake retried with the same `MessagePort`. A port can only be transferred
  once, so attempts 2–13 threw `Port at index 0 is already neutered` and the whole
  thing reported "the proxy iframe never answered".
- The init message lost its `type` field during a refactor, so the worker received a
  message it did not recognise and never started.
- The watchdog was shorter than the model download.
- `ensure()` could resolve to a marker string while the worker came up mid-call, and
  the caller tried to call it as a pipeline: `pipe is not a function`.
- A per-cue transcription error was written to the fatal `asrError` field, so one
  bad cue disabled ASR for the rest of the video.

## 6. Not verified

- **Firefox.** Everything above was measured on Chromium. `GM_addElement` and the
  `data:` iframe behave the same way in principle, but that is an assumption.
- **Live video and the worker in the same browser.** Each half was verified in a
  different browser: the archive and alignment where YouTube video plays, the worker
  under a real Tampermonkey. The combination was not.
- **`fmt=mp4a`.** Firefox may receive AAC in fMP4 rather than WebM/Opus.
  `decodeAudioData` on a fragmented MP4 slice was never tested.

## 7. Cost

- ~40 MB model, once, cached by the browser.
- Decoded PCM is held only while a cue still needs it, bounded to ~5 minutes behind
  the playhead.
- Raw audio bytes are capped at 96 MB and capture stops once every cue is answered.
