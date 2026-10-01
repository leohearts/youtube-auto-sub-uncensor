// ==UserScript==
// @name         YouTube Auto-Sub Uncensor
// @namespace    https://github.com/omp/youtube-auto-sub-uncensor
// @version      1.0.0
// @description  YouTube censors profanity in auto-generated captions server-side, replacing the word with "[ __ ]". This recovers the real word from the video's own audio with a local Whisper model running in a worker thread, and shows a grammar/timing guess until the transcript is ready.
// @author       omp
// @match        https://www.youtube.com/*
// @run-at       document-start
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_registerMenuCommand
// @grant        GM_unregisterMenuCommand
// @grant        GM_addElement
// @grant        unsafeWindow
// @connect      cdn.jsdelivr.net
// @connect      huggingface.co
// @connect      cdn-lfs.huggingface.co
// @connect      cdn-lfs-us-1.huggingface.co
// @connect      cdn-lfs-eu-1.huggingface.co
// @noframes
// ==/UserScript==

/*
 * Why this is built the way it is (verified against live YouTube, 2026-10):
 *
 * 1. The censorship happens on the server. The timedtext payload itself contains
 *    "[\u00a0__\u00a0]" (NBSP padded), not a display filter. Every format
 *    (json3/srv1/srv3/vtt/ttml) and every client param (c=ANDROID/IOS/TVHTML5/MWEB)
 *    returns byte-identical censored text, and tlang= is rate limited to 429. The
 *    original word is nowhere in YouTube's data, so it has to come from the audio.
 *
 * 2. The timedtext endpoint now requires a `pot` (proof-of-origin) token. Without it
 *    you get HTTP 200 with content-length: 0. The player appends
 *    "pot=...&potc=1&c=WEB&cver=...&cbr=...&cos=...&cplatform=DESKTOP" to its own
 *    request URL, so this script never builds a caption URL itself - it captures the
 *    one the player already uses.
 *
 * 3. Audio has to come from ahead of the playhead, or a correction can never arrive
 *    before the caption it belongs to. Recording the player's live output cannot do
 *    that. Instead the audio SourceBuffer is read directly: the player appends WebM/
 *    Opus to it well ahead of the playhead, each Cluster carries an absolute timecode
 *    in milliseconds, and `decodeAudioData` accepts both the whole stream and a single
 *    `header + cluster group` slice (~30 ms per 10 s of audio).
 *
 *    Ad breaks reuse the same pipeline and their Clusters also start near zero, so
 *    appends are ignored while an ad is showing and the archive resynchronises on the
 *    EBML header. Without that, ad audio lands on the video's timeline.
 *
 * 4. The censored word is picked out of the transcript by aligning it against the
 *    caption's own words. Matching by time does not work: YouTube's per-word tOffsetMs
 *    was measured 0.4 s out on a real cue, wider than the gap between words, which is
 *    how "slow as [ __ ]" came back as "as".
 *
 * 5. Inference has a floor of roughly 1.5 s - the mel spectrogram is always padded to
 *    30 s, so the encoder cost is fixed however short the clip is. On the main thread
 *    that is a visible freeze per censored word. youtube.com's CSP (`script-src` with
 *    no `blob:`) refuses a Worker built from the page, so the worker is built inside a
 *    `data:` iframe injected with GM_addElement, which does not carry the page's
 *    policy. Without GM_addElement that path fails and inference falls back to the
 *    main thread.
 */

(function () {
  'use strict';

  const W = (typeof unsafeWindow !== 'undefined' && unsafeWindow) ? unsafeWindow : window;

  // JS \s matches U+00A0, which is what YouTube uses inside the placeholder.
  const PLACEHOLDER = /\[\s*_+\s*\]/;
  const PLACEHOLDER_G = /\[\s*_+\s*\]/g;
  const WS_G = /[\s\u00a0]+/g;

  const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';
  const TRANSFORMERS_BASE = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/';

  const DEFAULTS = {
    enabled: true,
    asr: true,
    autoCaptions: true,
    modelEn: 'Xenova/whisper-tiny.en',
    modelOther: 'Xenova/whisper-tiny',
  };

  /* ------------------------------------------------------------------ *
   * Settings
   * ------------------------------------------------------------------ */

  const store = {
    read() {
      let raw = null;
      try { raw = GM_getValue('config', null); } catch (e) { /* no manager */ }
      if (typeof raw === 'string') { try { raw = JSON.parse(raw); } catch (e) { raw = null; } }
      return Object.assign({}, DEFAULTS, raw || {});
    },
    write(c) {
      try { GM_setValue('config', JSON.stringify(c)); } catch (e) { /* ignore */ }
    },
  };

  let config = store.read();

  // Box-filtered decimation to the 16 kHz mono that Whisper expects. Both audio paths
  // need it, so it lives in one place.
  function to16k(src, rate) {
    const ratio = rate / 16000;
    const outLen = Math.floor(src.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const s = Math.floor(i * ratio);
      const e = Math.min(src.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = s; j < e; j++) sum += src[j];
      out[i] = sum / (e - s);
    }
    return out;
  }

  /* ------------------------------------------------------------------ *
   * Shared state
   * ------------------------------------------------------------------ */

  const state = {
    videoId: null,
    lang: null,
    trackUrl: null,
    cues: [],           // { startMs, durMs, segs, text, norm, seq }
    normIndex: null,    // normalized cue text -> cue index, for O(1) DOM lookups
    corrections: null,  // Map<cueIndex, Map<segIndex, {word, source}>>
    asrWords: new Map(), // stable key -> word recovered by Whisper
    asrReady: false,
    asrLoading: false,
    asrWarming: false,
    asrError: null,       // load failure: fatal, do not retry
    asrLastError: null,   // last per-cue transcription error: retried, not fatal
    asrBlocked: null,   // set when captions were found but audio could not be captured
    pumpGate: null,     // why the ASR queue is not advancing right now
    queue: [],
    busy: false,
    pumping: false,
    pumpTimer: null,
    captionAttempt: 0,
    captionRetries: 0,
    parseError: null,   // surfaced in the pill: a parse failure means no cues at all
    pill: null,
  };

  function resetVideoState(videoId) {
    state.videoId = videoId;
    state.lang = null;
    state.trackUrl = null;
    state.cues = [];
    state.normIndex = null;
    state.corrections = new Map();
    state.asrWords = new Map();
    state.queue = [];
    state.pumping = false;
    if (state.pumpTimer) { clearTimeout(state.pumpTimer); state.pumpTimer = null; }
    state.asrError = null;
    state.asrLastError = null;
    state.asrBlocked = null;
    state.pumpGate = null;
    state.captionAttempt = 0;
    state.captionRetries = 0;
    tap.reset();
    // The archive is deliberately not reset here: it belongs to the media pipeline and
    // starts over when a new audio SourceBuffer is created.
    updatePill();
  }

  /* ------------------------------------------------------------------ *
   * Caption track capture
   *
   * The player's own request URL carries the `pot` token. We capture both the URL
   * and the response body, and only accept auto-generated tracks (kind=asr) that
   * are not machine-translated (no tlang=).
   * ------------------------------------------------------------------ */

  function isAutoCaptionUrl(url) {
    if (typeof url !== 'string') return false;
    if (url.indexOf('/api/timedtext') === -1) return false;
    if (url.indexOf('kind=asr') === -1) return false;
    if (/[?&]tlang=/.test(url)) return false;
    return true;
  }

  function onCaptionResponse(url, body) {
    if (!body || body.length < 32) return;
    // Pre-roll ads carry their own caption track; never let one replace the video's.
    const vid = /[?&]v=([^&]+)/.exec(url);
    if (state.videoId && (!vid || vid[1] !== state.videoId)) return;
    let data;
    try { data = JSON.parse(body); } catch (e) { return; }
    if (!data || !Array.isArray(data.events)) return;
    if (state.trackUrl === url && state.cues.length) return; // already have this one

    const langMatch = /[?&]lang=([^&]+)/.exec(url);
    state.trackUrl = url;
    state.lang = langMatch ? decodeURIComponent(langMatch[1]) : 'en';
    try {
      state.cues = parseTrack(data);
      state.corrections = new Map();
      analyze();
    } catch (e) {
      state.parseError = String(e && e.stack ? e.stack : e).slice(0, 400);
    }
    updatePill();
  }

  // Element/XHR expando properties do not survive Tampermonkey's DOM proxy, so
  // per-object state lives in WeakMaps instead.
  const xhrUrl = new WeakMap();
  const elState = new WeakMap();

  function installCaptureHooks() {
    const fetchFn = W.fetch;
    if (typeof fetchFn === 'function' && !fetchFn.__ytasu) {
      const patched = function (input, init) {
        const url = (typeof input === 'string') ? input : (input && input.url) || '';
        const promise = fetchFn.apply(this, arguments);
        if (isAutoCaptionUrl(url)) {
          promise.then(function (res) {
            try { res.clone().text().then(function (t) { onCaptionResponse(url, t); }, function () {}); } catch (e) {}
          }, function () {});
        }
        return promise;
      };
      patched.__ytasu = true;
      W.fetch = patched;
    }

    const XHR = W.XMLHttpRequest;
    if (XHR && XHR.prototype && !XHR.prototype.__ytasu) {
      const origOpen = XHR.prototype.open;
      const origSend = XHR.prototype.send;
      XHR.prototype.open = function (method, url) {
        xhrUrl.set(this, url);
        return origOpen.apply(this, arguments);
      };
      XHR.prototype.send = function () {
        const url = xhrUrl.get(this);
        if (isAutoCaptionUrl(url)) {
          const self = this;
          this.addEventListener('load', function () {
            try { onCaptionResponse(url, self.responseText); } catch (e) {}
          });
        }
        return origSend.apply(this, arguments);
      };
      XHR.prototype.__ytasu = true;
    }

    // The player's own buffered audio is the only audio that exists ahead of the
    // playhead. SourceBuffer.mimeType reads back empty on live YouTube, so the type is
    // remembered when the buffer is created rather than queried at append time.
    const MS = W.MediaSource;
    if (MS && MS.prototype && !MS.prototype.__ytasu) {
      const origAdd = MS.prototype.addSourceBuffer;
      MS.prototype.addSourceBuffer = function (mime) {
        const sb = origAdd.apply(this, arguments);
        const type = String(mime || '');
        if (/^audio\//i.test(type)) {
          // A new audio SourceBuffer means a new byte stream, so this - not the video
          // id - is where the archive starts over. Resetting on video change wiped the
          // EBML header while the same SourceBuffer kept appending, which left a
          // header-less stream that could never be decoded.
          archive.reset();
          archive.mime = type;
          try {
            const origAppend = sb.appendBuffer;
            sb.appendBuffer = function (data) {
              try {
                if (data && typeof data.byteLength === 'number') {
                  const u8 = new Uint8Array(data.buffer
                    ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
                    : data);
                  archive.append(u8, sb.timestampOffset || 0);
                  scheduleDecode();
                }
              } catch (e) {
                archive.error = String(e && e.message ? e.message : e);
              }
              return origAppend.apply(this, arguments);
            };
          } catch (e) { /* leave the buffer untouched */ }
        }
        return sb;
      };
      MS.prototype.__ytasu = true;
    }
  }

  /* ------------------------------------------------------------------ *
   * Track parsing + analysis
   * ------------------------------------------------------------------ */

  function parseTrack(data) {
    const cues = [];
    for (const ev of data.events) {
      if (!ev || !Array.isArray(ev.segs)) continue;
      const segs = ev.segs
        .filter(function (s) { return s && typeof s.utf8 === 'string'; })
        .map(function (s) {
          return { utf8: s.utf8, tOffsetMs: typeof s.tOffsetMs === 'number' ? s.tOffsetMs : null };
        });
      if (!segs.length) continue;
      const startMs = ev.tStartMs || 0;
      const durMs = ev.dDurationMs || 0;
      cues.push({ startMs: startMs, durMs: durMs, segs: segs, text: segs.map(function (s) { return s.utf8; }).join('') });
    }
    cues.sort(function (a, b) { return a.startMs - b.startMs; });
    for (const cue of cues) {
      cue.norm = normalize(cue.text);

      // Auto-captions skip non-speech, so the gap after a word can be seconds of
      // laughter or music rather than the word's own length. The median gap between
      // this cue's own words is the honest estimate of how long one word lasts here.
      const gaps = [];
      for (let i = 1; i < cue.segs.length; i++) {
        const a = cue.segs[i - 1].tOffsetMs;
        const b = cue.segs[i].tOffsetMs;
        if (a !== null && b !== null && b > a) gaps.push(b - a);
      }
      gaps.sort(function (x, y) { return x - y; });
      const median = gaps.length ? gaps[Math.floor(gaps.length / 2)] : 0;
      const typical = Math.min(1200, Math.max(250, median || 350));

      // Fill in missing offsets so every seg has a start and an end.
      let cursor = 0;
      for (let i = 0; i < cue.segs.length; i++) {
        const seg = cue.segs[i];
        seg.startOffset = (seg.tOffsetMs === null) ? cursor : seg.tOffsetMs;
        const next = cue.segs[i + 1];
        let end;
        if (next && next.tOffsetMs !== null) {
          end = next.tOffsetMs;
          seg.rawEndOffset = end;
          // A gap far larger than this cue's rhythm means the next word is simply far
          // away in the audio, not that this word is long. Cap it so the censored
          // word can be transcribed as soon as it is actually spoken.
          if (end - seg.startOffset > typical * 4) end = seg.startOffset + typical * 2;
        } else {
          end = seg.startOffset + typical * 2;
          seg.rawEndOffset = Math.max(cue.durMs, end);
        }
        seg.endOffset = Math.max(seg.startOffset + 150, Math.round(end));
        cursor = seg.endOffset;
      }
    }
    return cues;
  }

  function normalize(s) {
    return s.replace(PLACEHOLDER_G, '\u0000').replace(WS_G, ' ').trim().toLowerCase();
  }

  // ASR results are kept separately, keyed by where the word sits in the video rather
  // than by index. The player refetches the caption track with a fresh URL whenever it
  // reinitialises (quality change, some seeks), and re-parsing rebuilds every cue and
  // every guess - without this, a re-parse silently throws away everything Whisper had
  // already worked out and the guesses come back.
  function wordKey(cue, seg) {
    return cue.startMs + '/' + seg.startOffset;
  }

  // Strips what wraps a word so two spellings of it can be compared. Used for both the
  // caption's words and Whisper's - they must agree, or an apostrophe would stop a word
  // from ever matching and the alignment would silently lose that anchor.
  function stripWord(s) {
    return s.replace(/^[^A-Za-z']+/, '').replace(/[^A-Za-z']+$/, '').toLowerCase();
  }

  // Flatten every cue into a word list so a placeholder can see its neighbours
  // even when they sit in a different seg or cue. Tokens belonging to the
  // placeholder itself are skipped, otherwise "[ __ ]".split() would put "[", "__"
  // and "]" next to the gap and every neighbour lookup would be off by three.
  //
  // Each cue also keeps its own ordered word sequence with the gaps marked. That
  // sequence is what the transcript gets aligned against: the words around a censored
  // word are known exactly, so the censored word can be read off by position instead
  // of by time.
  function analyze() {
    state.normIndex = null;
    const words = [];
    const gaps = [];
    state.cues.forEach(function (cue, cueIdx) {
      const seq = [];
      cue.segs.forEach(function (seg, segIdx) {
        if (PLACEHOLDER.test(seg.utf8)) {
          seg.placeholder = true;
          seg.cueIdx = cueIdx;
          seg.segIdx = segIdx;
          seg.seqIndex = seq.length;
          gaps.push({ seg: seg, anchor: words.length, cueIdx: cueIdx, seqIndex: seq.length });
          seq.push({ word: '', gap: true, segIdx: segIdx });
          return;
        }
        seg.utf8.split(/\s+/).forEach(function (tok) {
          if (!tok) return;
          const w = stripWord(tok);
          words.push(w);
          seq.push({ word: w, gap: false, segIdx: segIdx });
        });
      });
      cue.seq = seq;
    });

    gaps.forEach(function (gap) {
      const a = gap.anchor;
      gap.seg.context = {
        prev1: a > 0 ? words[a - 1] : '',
        prev2: a > 1 ? words[a - 2] : '',
        next1: a < words.length ? words[a] : '',
        next2: a + 1 < words.length ? words[a + 1] : '',
        durMs: Math.max(0, gap.seg.endOffset - gap.seg.startOffset),
      };
    });

    // Apply the instant guess to every placeholder right away, unless Whisper already
    // answered this exact word in an earlier parse.
    state.cues.forEach(function (cue, cueIdx) {
      const map = new Map();
      cue.segs.forEach(function (seg, segIdx) {
        if (!seg.placeholder) return;
        const remembered = state.asrWords.get(wordKey(cue, seg));
        map.set(segIdx, remembered
          ? { word: remembered, source: 'asr' }
          : { word: guessWord(seg.context), source: 'guess' });
      });
      if (map.size) state.corrections.set(cueIdx, map);
    });

    // Every censored word becomes a job. The list is never consumed: jobs are marked
    // done instead, so seeking back to a part of the video that was skipped (or played
    // before the tap was ready) makes those cues transcribable again.
    state.queue = [];
    state.cues.forEach(function (cue, cueIdx) {
      cue.segs.forEach(function (seg, segIdx) {
        if (!seg.placeholder) return;
        state.queue.push({
          cueIdx: cueIdx,
          segIdx: segIdx,
          seqIndex: seg.seqIndex,
          key: wordKey(cue, seg),
          targetFrom: (cue.startMs + seg.startOffset) / 1000,
          targetTo: (cue.startMs + seg.endOffset) / 1000,
          // The slice reaches past the estimated end so a drawn-out word still has its
          // tail in the audio, while readiness only waits for the estimated end.
          sliceFrom: (cue.startMs + seg.startOffset) / 1000 - 4,
          sliceTo: (cue.startMs + Math.min(seg.rawEndOffset, seg.endOffset + 2500)) / 1000 + 0.5,
          done: state.asrWords.has(wordKey(cue, seg)),
          tries: 0,
          audioMisses: 0,
        });
      });
    });
    state.queue.sort(function (a, b) { return a.targetTo - b.targetTo; });

    // Caption lookups run on every DOM mutation, so exact matches are indexed instead
    // of scanned: a long video has tens of thousands of cues.
    state.normIndex = new Map();
    state.cues.forEach(function (cue, i) {
      if (cue.norm && !state.normIndex.has(cue.norm)) state.normIndex.set(cue.norm, i);
    });
  }

  /* ------------------------------------------------------------------ *
   * Heuristic guess
   *
   * This is only a stand-in shown until Whisper finishes. It is a guess, and it is
   * wrong sometimes ("slow as [ ]" is "shit", not "fuck"). Grammar decides between
   * the intensifier ("fucking") and the standalone word ("fuck"); when grammar is
   * ambiguous the word's measured audio length decides.
   * ------------------------------------------------------------------ */

  const NOUNS = new Set([
    'luck', 'idea', 'break', 'clue', 'way', 'thing', 'time', 'day', 'mess', 'idiot',
    'jerk', 'moron', 'bastard', 'guy', 'dude', 'ass', 'face', 'head', 'hell',
    'mistake', 'problem', 'reason', 'point', 'chance', 'sense', 'difference',
  ]);
  const ADJECTIVES = new Set([
    'sick', 'good', 'bad', 'great', 'cool', 'beautiful', 'tired', 'hard', 'stupid',
    'crazy', 'amazing', 'terrible', 'awesome', 'insane', 'slow', 'fast', 'dumb',
    'ugly', 'nice', 'funny', 'serious', 'big', 'small', 'loud', 'quiet', 'hot',
    'cold', 'weird', 'scary', 'easy', 'simple', 'perfect', 'horrible', 'awful',
    'wild', 'ridiculous', 'impressive', 'expensive', 'cheap', 'heavy', 'sharp',
    'dangerous', 'fun', 'boring', 'confusing', 'annoying', 'embarrassing', 'late',
    'early', 'long', 'short', 'deep', 'strong', 'weak', 'sweet', 'bright', 'dark',
    'smooth', 'rough', 'soft', 'clean', 'dirty', 'full', 'empty', 'alive', 'dead',
    'safe', 'useful', 'important', 'complex', 'honest', 'rude', 'kind', 'mean',
    'brave', 'lazy', 'smart', 'clever', 'wise', 'silly', 'lucky', 'proud', 'rich',
    'poor', 'young', 'old', 'new', 'fresh', 'real', 'actual', 'absolute',
  ]);
  const ADVERBS_BEFORE = new Set([
    'so', 'just', 'still', 'absolutely', 'really', 'totally', 'completely',
    'pretty', 'literally', 'actually', 'super', 'too', 'very',
  ]);

  function guessWord(ctx) {
    if (!ctx) return 'fucking';
    const p1 = ctx.prev1, p2 = ctx.prev2, n1 = ctx.next1;

    if (n1 === 'you') return 'fuck';
    if (p1 === 'the' && p2 === 'shut') return 'fuck';
    if (n1 === 'off') return 'fucked';
    if (p1 === 'as' || p1 === 'about') return 'fuck';
    if ((p1 === 'a' || p1 === 'an') && n1 && !NOUNS.has(n1) && !ADJECTIVES.has(n1)) return 'fuck';
    if (p1 === 'a' || p1 === 'an') return 'fucking';
    if (p1 === 'no' || p1 === 'any') return 'fucking';
    if (ADVERBS_BEFORE.has(p1)) return 'fucking';
    if (n1 && ADJECTIVES.has(n1)) return 'fucking';
    if (p1 === 'shut' && n1 === 'up') return 'fuck';
    // Nothing conclusive: fall back to how long the word actually took.
    return ctx.durMs <= 320 ? 'fuck' : 'fucking';
  }

  /* ------------------------------------------------------------------ *
   * Audio archive - reads the audio the player has already buffered
   *
   * Recording the player's live output can only ever produce audio the playhead has
   * already passed, so a correction can never arrive before the caption it belongs
   * to. The player, however, appends media to its SourceBuffers well ahead of the
   * playhead, and those bytes are ours to read.
   *
   * Verified against live YouTube: the audio SourceBuffer is WebM/Opus, each Cluster
   * carries an absolute timecode in milliseconds, and `decodeAudioData` accepts both
   * the whole stream and a single `header + cluster group` slice (~30 ms per 10 s of
   * audio). That makes it possible to decode a censored word's audio, transcribe it
   * and have the answer ready before the caption is ever drawn.
   * ------------------------------------------------------------------ */

  const ARCHIVE_MAX_BYTES = 96 * 1024 * 1024;   // raw WebM kept before capture stops
  const ARCHIVE_KEEP_BEHIND_SEC = 300;          // decoded PCM kept behind the playhead

  const archive = {
    buf: null,            // growing copy of the audio byte stream
    len: 0,
    headerEnd: -1,        // byte offset of the first Cluster; everything before it is the header
    clusters: [],         // { byte, tsOffsetMs, tcMs, decoded } ascending by byte
    scanFrom: 0,
    ctx: null,
    pcm: [],              // { startSec, data } at 16 kHz mono, ascending by startSec
    decodedThroughMs: -Infinity,
    decoding: false,
    decodeFailures: 0,
    error: null,
    synced: false,        // an EBML header has been seen, so the buffer is a real stream
    full: false,          // the raw buffer hit its cap and stopped growing
    mime: null,           // the audio SourceBuffer's type, for diagnostics
    bytes: 0,

    reset: function () {
      this.buf = null;
      this.len = 0;
      this.headerEnd = -1;
      this.clusters = [];
      this.scanFrom = 0;
      this.pcm = [];
      this.decodedThroughMs = -Infinity;
      this.decoding = false;
      this.decodeFailures = 0;
      this.error = null;
      this.synced = false;
      this.full = false;
      this.bytes = 0;
    },

    active: function () {
      return this.synced && this.len > 0 && this.headerEnd >= 0;
    },

    // An ad plays through the same kind of pipeline and its Clusters carry their own
    // timecodes starting near zero, so capturing it would splice two unrelated videos
    // onto one timeline. That is exactly how "we plan the whole area around this
    // defining feature, the water" ended up answering a cue from this video.
    inAd: function () {
      try {
        return !!document.querySelector('.ad-showing');
      } catch (e) {
        return false;
      }
    },

    append: function (bytes, tsOffsetSec) {
      if (this.inAd()) return;
      // Resynchronise on the EBML header rather than trusting append order: it marks
      // the true start of a byte stream, so a reused SourceBuffer or a stream that
      // changed underneath us cannot leave a header-less buffer behind. Starting a
      // video from the middle lands here too - the player appends the init segment
      // before the first media range it fetched for the seek.
      const isHeader = bytes.length >= 4 && bytes[0] === 0x1A && bytes[1] === 0x45 &&
                       bytes[2] === 0xDF && bytes[3] === 0xA3;
      if (isHeader) this.reset();
      else if (!this.synced) return;   // nothing decodable until a header shows up

      // Once the captions are known and every censored word has been answered there is
      // nothing left to serve, so stop growing the copy.
      if (state.cues.length && !hasPendingJobs()) return;
      this.synced = true;
      this.bytes += bytes.length;
      if (this.full) return;
      if (this.len + bytes.length > ARCHIVE_MAX_BYTES) {
        this.full = true;
        return;
      }
      if (!this.buf) this.buf = new Uint8Array(Math.max(1 << 16, bytes.length * 2));
      if (this.len + bytes.length > this.buf.length) {
        let cap = this.buf.length;
        while (cap < this.len + bytes.length) cap *= 2;
        const next = new Uint8Array(cap);
        next.set(this.buf.subarray(0, this.len));
        this.buf = next;
      }
      this.buf.set(bytes, this.len);
      this.len += bytes.length;
      this.scanClusters(tsOffsetSec || 0);
    },

    // The Cluster ID (1F 43 B6 75) can straddle two appends, so the scan restarts a
    // few bytes behind the new data. A Cluster usually begins exactly at the first
    // byte of a new append, so that byte must not be skipped.
    scanClusters: function (tsOffsetSec) {
      const buf = this.buf;
      const start = Math.max(0, this.scanFrom - 3);
      const tsOffsetMs = Math.round((tsOffsetSec || 0) * 1000);
      for (let i = start; i + 4 <= this.len; i++) {
        if (buf[i] !== 0x1F || buf[i + 1] !== 0x43 || buf[i + 2] !== 0xB6 || buf[i + 3] !== 0x75) continue;
        const last = this.clusters[this.clusters.length - 1];
        if (last && last.byte >= i) continue;
        this.clusters.push({ byte: i, tsOffsetMs: tsOffsetMs, tcMs: null, decoded: false });
        if (this.headerEnd < 0) this.headerEnd = i;
      }
      this.scanFrom = this.len;
    },

    clusterTimecodeMs: function (offset) {
      const buf = this.buf;
      const idLen = 4;
      const size = readVint(buf, offset + idLen);
      if (!size) return null;
      let pos = offset + idLen + size.len;
      const end = Math.min(this.len, pos + size.val);
      while (pos < end - 2) {
        const id = readVint(buf, pos);
        if (!id) return null;
        let idVal = 0;
        for (let k = 0; k < id.len; k++) idVal = idVal * 256 + buf[pos + k];
        const sz = readVint(buf, pos + id.len);
        if (!sz) return null;
        const data = pos + id.len + sz.len;
        if (idVal === 0xE7) {
          let tc = 0;
          for (let k = 0; k < sz.val; k++) tc = tc * 256 + buf[data + k];
          return tc;
        }
        pos = data + sz.val;
      }
      return null;
    },

    // MSE defines presentation time as the container timestamp plus the SourceBuffer's
    // timestampOffset at the time of the append. Reading only the container timestamp
    // happens to work when playback starts at zero and silently skews everything when
    // the player offsets a range it fetched for a seek.
    timeMsAt: function (cluster) {
      if (cluster.tcMs === null) {
        const raw = this.clusterTimecodeMs(cluster.byte);
        if (raw === null) return null;
        cluster.tcMs = raw;
      }
      return cluster.tcMs + cluster.tsOffsetMs;
    },

    // The next cluster that has not been decoded and has a successor to bound it.
    // Tracked per cluster rather than with a high-water mark: seeking backwards makes
    // the player append an earlier range, and a watermark would skip it forever.
    nextGroup: function () {
      if (this.headerEnd < 0) return null;
      for (let i = 0; i < this.clusters.length - 1; i++) {
        const c = this.clusters[i];
        if (c.decoded) continue;
        const t = this.timeMsAt(c);
        if (t === null) continue;
        return { index: i, from: c.byte, to: this.clusters[i + 1].byte, startMs: t };
      }
      return null;
    },

    bytesFor: function (group) {
      const out = new Uint8Array(this.headerEnd + (group.to - group.from));
      out.set(this.buf.subarray(0, this.headerEnd), 0);
      out.set(this.buf.subarray(group.from, group.to), this.headerEnd);
      return out;
    },

    decodeLoop: async function () {
      if (this.decoding || !this.active() || this.decodeFailures >= 3) return;
      this.decoding = true;
      try {
        const AC = W.AudioContext || W.webkitAudioContext;
        if (!this.ctx && AC) this.ctx = new AC();
        if (!this.ctx) return;
        for (let guard = 0; guard < 8; guard++) {
          const group = this.nextGroup();
          if (!group) break;
          const bytes = this.bytesFor(group);
          let decoded = null;
          try {
            decoded = await this.ctx.decodeAudioData(bytes.buffer.slice(0));
          } catch (e) {
            // Mark it consumed so one bad slice cannot block the queue forever, and
            // give up after a few rather than retrying a format we cannot decode.
            this.clusters[group.index].decoded = true;
            this.decodeFailures++;
            this.error = 'decode failed at ' + Math.round(group.startMs / 1000) + 's: ' + (e && e.message ? e.message : e);
            continue;
          }
          this.clusters[group.index].decoded = true;
          this.pushPcm(group.startMs, decoded);
        }
      } finally {
        this.decoding = false;
      }
      if (hasPendingJobs()) scheduleDecode();
    },

    pushPcm: function (startMs, audioBuffer) {
      const out = to16k(audioBuffer.getChannelData(0), audioBuffer.sampleRate);
      const startSec = startMs / 1000;
      // Insert in time order: after a seek the player appends ranges out of order, and
      // a single list sorted by time keeps window() and covers() honest.
      let at = this.pcm.length;
      while (at > 0 && this.pcm[at - 1].startSec > startSec) at--;
      this.pcm.splice(at, 0, { startSec: startSec, data: out });
      const endMs = startMs + out.length / 16;
      if (endMs > this.decodedThroughMs) this.decodedThroughMs = endMs;
      this.trim();
    },

    // Decoded audio is only worth keeping while a cue that still needs it is in reach.
    // A cue far behind the playhead can never be served again: either it was already
    // transcribed, or the player no longer holds its audio to append. Without this
    // bound, starting in the middle of a video pins the oldest pending cue near zero
    // and the PCM grows for the whole session.
    trim: function () {
      const video = getVideo();
      const nowSec = video ? video.currentTime : 0;
      let keepFrom = nowSec - ARCHIVE_KEEP_BEHIND_SEC;
      for (let i = 0; i < state.queue.length; i++) {
        const job = state.queue[i];
        if (job.done) continue;
        if (job.sliceTo < nowSec - ARCHIVE_KEEP_BEHIND_SEC) continue;
        if (job.sliceFrom < keepFrom) keepFrom = job.sliceFrom;
      }
      while (this.pcm.length > 1 &&
             this.pcm[0].startSec + this.pcm[0].data.length / 16000 < keepFrom) {
        this.pcm.shift();
      }
    },

    // True once the decoded audio covers [reqFrom, reqTo] with no hole in between.
    // Allocation-free, because the pump asks this about every pending cue.
    coversRange: function (reqFrom, reqTo) {
      let cursor = reqFrom;
      for (let i = 0; i < this.pcm.length; i++) {
        const c = this.pcm[i];
        const cEnd = c.startSec + c.data.length / 16000;
        if (cEnd <= reqFrom) continue;
        if (c.startSec > cursor + 0.2) return false;   // hole before the span is covered
        if (cEnd > cursor) cursor = cEnd;
        if (cursor >= reqTo) return true;
      }
      return false;
    },

    window: function (fromSec, toSec, requiredFrom, requiredTo) {
      if (!this.pcm.length) return null;
      const picked = [];
      for (const c of this.pcm) {
        const cEnd = c.startSec + c.data.length / 16000;
        if (cEnd < fromSec || c.startSec > toSec) continue;
        picked.push(c);
      }
      if (!picked.length) return null;

      // The concatenation below assumes the pieces are back to back. If the player
      // appended a disjoint range there is a hole, and splicing across it would shift
      // every later timestamp - which is exactly what makes the wrong word get chosen.
      for (let i = 1; i < picked.length; i++) {
        const prevEnd = picked[i - 1].startSec + picked[i - 1].data.length / 16000;
        if (picked[i].startSec > prevEnd + 0.2) return null;
      }

      const firstT = picked[0].startSec;
      const last = picked[picked.length - 1];
      const lastT = last.startSec + last.data.length / 16000;
      const reqFrom = (requiredFrom === undefined) ? firstT : requiredFrom;
      const reqTo = (requiredTo === undefined) ? lastT : requiredTo;
      if (firstT > reqFrom + 0.2 || lastT < reqTo - 0.2) return null;

      const total = picked.reduce(function (a, c) { return a + c.data.length; }, 0);
      const out = new Float32Array(total);
      let off = 0;
      for (const c of picked) { out.set(c.data, off); off += c.data.length; }
      return { pcm: out, startSec: firstT, endSec: lastT, pieces: picked.length };
    },
  };

  // Minimal EBML variable-length integer reader: the length is encoded in the leading
  // zero bits, and for sizes the marker bit is not part of the value.
  function readVint(buf, i) {
    if (i >= buf.length) return null;
    const b = buf[i];
    let len = 0;
    if (b & 0x80) len = 1;
    else if (b & 0x40) len = 2;
    else if (b & 0x20) len = 3;
    else if (b & 0x10) len = 4;
    else return null;
    if (i + len > buf.length) return null;
    let val = b & (0xFF >> len);
    for (let k = 1; k < len; k++) val = val * 256 + buf[i + k];
    return { len: len, val: val };
  }

  let decodeTimer = null;
  function scheduleDecode() {
    if (decodeTimer) return;
    decodeTimer = setTimeout(function () {
      decodeTimer = null;
      archive.decodeLoop();
    }, 120);
  }

  /* ------------------------------------------------------------------ *
   * Audio tap (fallback when the media pipeline is not MSE)
   *
   * video.captureStream() is used instead of createMediaElementSource() so the
   * element's own audio routing is never taken over. A ScriptProcessorNode feeds a
   * rolling buffer of mono PCM, timestamped against video.currentTime.
   *
   * This can only ever produce audio the playhead has already passed, so corrections
   * from this path always land late. The archive above is the path that matters.
   * ------------------------------------------------------------------ */

  const RING_SECONDS = 30;

  // createMediaElementSource may only be called once per element, so the node is kept
  // and reused; the destination link is remembered so re-attaching cannot double it.
  const mediaElementSources = new WeakMap();
  const routedToDestination = new WeakSet();

  const tap = {
    ctx: null,
    attached: null,
    node: null,
    sink: null,
    stream: null,
    source: null,
    mode: null,        // 'captureStream' | 'mediaElementSource'
    reason: null,      // why capture is not producing audio
    frames: 0,         // onaudioprocess callbacks seen since attach
    attachedAt: 0,
    fallbackTried: false,
    chunks: [],        // { tStart, data }
    anchor: null,      // { t, n }
    total: 0,

    reset: function () {
      this.chunks = [];
      this.anchor = null;
      this.total = 0;
    },

    detach: function () {
      try { if (this.node) this.node.disconnect(); } catch (e) {}
      try { if (this.sink) this.sink.disconnect(); } catch (e) {}
      if (this.seekVideo && this.onSeeked) {
        try { this.seekVideo.removeEventListener('seeked', this.onSeeked); } catch (e) {}
      }
      this.seekVideo = null;
      // A source built by createMediaElementSource owns the element's audio routing;
      // disconnecting it would leave the player silent, so it is left connected.
      if (this.mode !== 'mediaElementSource') {
        try { if (this.source) this.source.disconnect(); } catch (e) {}
      }
      try { if (this.stream) this.stream.getTracks().forEach(function (t) { t.stop(); }); } catch (e) {}
      this.node = null;
      this.sink = null;
      this.stream = null;
      this.source = null;
      this.attached = null;
      this.mode = null;
      this.frames = 0;
      this.reset();
    },

    ensureContext: function () {
      const AC = W.AudioContext || W.webkitAudioContext;
      if (!AC) { this.reason = 'no AudioContext in this browser'; return null; }
      try {
        if (!this.ctx || this.ctx.state === 'closed') this.ctx = new AC();
      } catch (e) {
        this.reason = 'AudioContext failed: ' + e.message;
        return null;
      }
      if (this.ctx.state === 'suspended') {
        const self = this;
        this.ctx.resume().catch(function () {});
        // Firefox keeps a context created before a gesture suspended until one happens.
        setTimeout(function () { if (self.ctx && self.ctx.state === 'suspended') self.reason = 'AudioContext is suspended (play the video once to allow audio)'; }, 1200);
      }
      return this.ctx;
    },

    attach: function (video, forceElementSource) {
      if (!forceElementSource && this.attached === video && this.node) return true;
      const sameVideo = this.attached === video;
      this.detach();
      const ctx = this.ensureContext();
      if (!ctx) return false;
      if (!video) { this.reason = 'no video element'; return false; }
      if (!sameVideo) this.reason = null;

      let source = null;

      // captureStream() is preferred: it leaves the element's own audio routing alone.
      if (!forceElementSource && typeof video.captureStream === 'function') {
        try {
          const stream = video.captureStream();
          this.stream = stream;
          // Firefox can hand back the stream before its audio track appears, so an
          // empty track list here is not fatal - a source node over a stream that
          // gains a track later still delivers audio.
          source = ctx.createMediaStreamSource(stream);
          this.mode = 'captureStream';
          if (!stream.getAudioTracks().length) this.reason = 'captureStream gave no audio track yet';
        } catch (e) {
          this.reason = 'captureStream failed: ' + e.message;
          source = null;
        }
      } else if (forceElementSource) {
        this.reason = 'captureStream produced no audio';
      } else {
        this.reason = 'captureStream is not available';
      }

      if (!source) {
        // createMediaElementSource always carries the element's audio, at the cost of
        // owning its routing. It may only be called once per element, so it is cached.
        source = mediaElementSources.get(video) || null;
        if (source) {
          this.mode = 'mediaElementSource';
          this.reason = null;
        } else {
          try {
            source = ctx.createMediaElementSource(video);
            mediaElementSources.set(video, source);
            this.mode = 'mediaElementSource';
            this.reason = null;
          } catch (e) {
            this.reason = (this.reason ? this.reason + '; ' : '') + 'createMediaElementSource failed: ' + e.message;
            return false;
          }
        }
      }

      try {
        const node = ctx.createScriptProcessor(4096, 1, 1);
        const sink = ctx.createGain();
        sink.gain.value = 0;
        source.connect(node);
        node.connect(sink);
        sink.connect(ctx.destination);
        // Keep the element audible when we took over its routing.
        if (this.mode === 'mediaElementSource' && !routedToDestination.has(video)) {
          source.connect(ctx.destination);
          routedToDestination.add(video);
        }
        const self = this;
        node.onaudioprocess = function (ev) { self.onAudio(ev, video); };
        // A seek really does invalidate everything buffered: the samples belong to the
        // old position and would be spliced onto the new one.
        if (!this.onSeeked) {
          this.onSeeked = function () {
            self.chunks = [];
            self.anchor = { t: video.currentTime, n: self.total };
          };
        }
        video.removeEventListener('seeked', this.onSeeked);
        video.addEventListener('seeked', this.onSeeked);
        this.seekVideo = video;
        this.source = source;
        this.node = node;
        this.sink = sink;
        this.attached = video;
        this.attachedAt = Date.now();
        this.frames = 0;
        this.reset();
        this.anchor = { t: video.currentTime, n: 0 };
        return true;
      } catch (e) {
        this.reason = 'audio graph failed: ' + e.message;
        this.detach();
        return false;
      }
    },

    onAudio: function (ev, video) {
      // While the player is paused the graph still pulls, but everything it delivers
      // is silence and video.currentTime is frozen - buffering it would map a pile of
      // chunks onto one instant and poison the slice for the next real cue.
      if (video.paused) return;
      this.frames++;

      const rate = this.ctx.sampleRate;
      const input = ev.inputBuffer.getChannelData(0);
      const copy = new Float32Array(input.length);
      copy.set(input);

      const now = video.currentTime;
      if (this.anchor) {
        const expected = this.anchor.t + (this.total - this.anchor.n) / rate;
        if (Math.abs(now - expected) > 1.2) {
          // Playback stopped and resumed, or the clock ran away from the audio. Only
          // re-anchor: audio already in the buffer still belongs to the times it was
          // stamped with, and dropping it here used to wipe the buffer on every pause.
          this.anchor = { t: now, n: this.total };
        }
      } else {
        this.anchor = { t: now, n: this.total };
      }
      const tStart = this.anchor.t + (this.total - this.anchor.n) / rate;
      this.total += copy.length;

      this.chunks.push({ tStart: tStart, data: copy });
      const cutoff = tStart - RING_SECONDS;
      while (this.chunks.length && this.chunks[0].tStart + this.chunks[0].data.length / rate < cutoff) {
        this.chunks.shift();
      }
    },

    // Video-time up to which the buffer actually holds audio. This is what decides
    // whether a censored word can be transcribed yet, and it leads the 100 ms tick.
    coveredTo: function () {
      if (!this.ctx || !this.chunks.length) return -Infinity;
      const last = this.chunks[this.chunks.length - 1];
      return last.tStart + last.data.length / this.ctx.sampleRate;
    },

    // Returns mono 16 kHz PCM covering [fromSec, toSec].
    //
    // [requiredFrom, requiredTo] must be present without a hole. Checking only the
    // first-to-last span hides a gap in the middle: a stalled tap or a buffer reset
    // leaves two islands of audio that would be spliced into a sentence nobody spoke.
    // A drawn-out word is exactly where that goes wrong, because the missing part is
    // the middle of the word.
    slice16k: function (fromSec, toSec, requiredFrom, requiredTo) {
      const rate = this.ctx ? this.ctx.sampleRate : 0;
      if (!rate || !this.chunks.length) return null;

      const picked = [];
      for (const chunk of this.chunks) {
        const cStart = chunk.tStart;
        const cEnd = cStart + chunk.data.length / rate;
        if (cEnd < fromSec || cStart > toSec) continue;
        picked.push(chunk);
      }
      if (!picked.length) return null;
      picked.sort(function (a, b) { return a.tStart - b.tStart; });

      const first = picked[0];
      const firstT = first.tStart;
      const last = picked[picked.length - 1];
      const lastT = last.tStart + last.data.length / rate;

      const reqFrom = (requiredFrom === undefined) ? firstT : requiredFrom;
      const reqTo = (requiredTo === undefined) ? lastT : requiredTo;
      let cursor = reqFrom;
      for (const c of picked) {
        const cStart = c.tStart;
        const cEnd = cStart + c.data.length / rate;
        if (cEnd <= reqFrom) continue;
        if (cStart > cursor + 0.2) return null;   // hole inside the span that matters
        if (cEnd > cursor) cursor = cEnd;
        if (cursor >= reqTo) break;
      }
      if (cursor < reqTo) return null;
      if (lastT - firstT < 0.4) return null;

      const total = picked.reduce(function (a, c) { return a + c.data.length; }, 0);
      const joined = new Float32Array(total);
      let offset = 0;
      for (const c of picked) { joined.set(c.data, offset); offset += c.data.length; }
      return { pcm: to16k(joined, rate), startSec: firstT, endSec: lastT, pieces: picked.length };
    },
  };

  /* ------------------------------------------------------------------ *
   * Local ASR
   * ------------------------------------------------------------------ */

  // Evaluating the bundle has to work in two very different places: a Tampermonkey
  // sandbox (where `eval` is the sandbox's own) and the raw page (where YouTube's
  // `require-trusted-types-for 'script'` also gates eval). Try both, in order.
  function evalClassic(code) {
    const attempts = [];
    const tt = W.trustedTypes;
    if (tt && typeof tt.createPolicy === 'function') {
      // A policy is creatable because YouTube's CSP has no `trusted-types` directive
      // restricting policy names.
      try {
        const policy = tt.createPolicy('ytasu' + Date.now(), { createScript: function (s) { return s; } });
        attempts.push(function () { return W.eval(policy.createScript(code)); });
      } catch (e) { /* fall through */ }
    }
    attempts.push(function () { return W.eval(code); });
    attempts.push(function () { return (0, eval)(code); });

    let lastError = null;
    for (const attempt of attempts) {
      try { attempt(); return; } catch (e) { lastError = e; }
    }
    throw lastError;
  }

  // The published build is ESM with a single trailing export and no imports, so it can
  // be turned into a classic script and evaluated where a module cannot load. Both the
  // main thread and the worker need exactly this, so it is produced once here.
  async function fetchTransformersSource() {
    if (W.__ytasuTransformersSource) return W.__ytasuTransformersSource;
    const res = await W.fetch(TRANSFORMERS_URL);
    const src = await res.text();
    const m = src.match(/export\s*\{([\s\S]*)\}\s*;?\s*(?:\/\/# sourceMappingURL=[^\n]*)?\s*$/);
    if (!m) throw new Error('unexpected transformers bundle layout');
    const pairs = m[1].split(',').map(function (s) { return s.trim(); }).filter(Boolean).map(function (s) {
      const p = s.split(/\s+as\s+/);
      return p.length === 2 ? JSON.stringify(p[1]) + ':' + p[0] : JSON.stringify(p[0]) + ':' + p[0];
    });
    const classic = src.slice(0, m.index) + 'globalThis.transformers={' + pairs.join(',') + '};';
    W.__ytasuTransformersSource = classic;
    return classic;
  }

  function configureTransformers(T, wasmBase) {
    T.env.allowLocalModels = false;
    T.env.backends.onnx.wasm.wasmPaths = wasmBase;
    // No cross-origin isolation on youtube.com, so the threaded backend is unavailable.
    T.env.backends.onnx.wasm.numThreads = 1;
    return T;
  }

  async function loadTransformers() {
    if (W.__ytasuTransformers) return W.__ytasuTransformers;
    evalClassic(await fetchTransformersSource());
    const T = W.transformers
      || (typeof globalThis !== 'undefined' && globalThis.transformers)
      || (typeof window !== 'undefined' && window.transformers);
    if (!T || typeof T.pipeline !== 'function') throw new Error('transformers failed to initialise');
    configureTransformers(T, TRANSFORMERS_BASE);
    W.__ytasuTransformers = T;
    return T;
  }

  // Runs inside the worker. Kept free of closures so it can be stringified.
  function workerMain() {
    var pipe = null;

    function evalClassic(code) {
      try { (0, eval)(code); return; } catch (e) {
        var tt = self.trustedTypes;
        if (tt && tt.createPolicy) {
          var policy = tt.createPolicy('ytasuW' + Date.now(), { createScript: function (s) { return s; } });
          (0, eval)(policy.createScript(code));
          return;
        }
        throw e;
      }
    }

    self.onmessage = async function (e) {
      var msg = e.data || {};
      self.postMessage({ type: 'progress', step: 'message:' + (msg.type || '?') });
      if (msg.type === 'init') {
        try {
          self.postMessage({ type: 'progress', step: 'eval:' + msg.classicSource.length });
          evalClassic(msg.classicSource);
          var T = self.transformers;
          T.env.allowLocalModels = false;
          T.env.backends.onnx.wasm.wasmPaths = msg.wasmBase;
          T.env.backends.onnx.wasm.numThreads = 1;
          self.postMessage({ type: 'progress', step: 'building pipeline' });
          pipe = await T.pipeline('automatic-speech-recognition', msg.model);
          self.postMessage({ type: 'progress', step: 'warming' });
          // Warm up here too, so the first real cue is not the slow one.
          await pipe(new Float32Array(16000 * 5), { return_timestamps: 'word' });
          self.postMessage({ type: 'progress', step: 'warm' });
          self.postMessage({ type: 'ready' });
        } catch (err) {
          self.postMessage({ type: 'init-error', error: String(err && err.message ? err.message : err) });
        }
        return;
      }
      if (msg.type === 'transcribe') {
        if (!pipe) { self.postMessage({ type: 'result', id: msg.id, error: 'not ready' }); return; }
        try {
          var res = await pipe(new Float32Array(msg.pcm), { return_timestamps: 'word' });
          self.postMessage({ type: 'result', id: msg.id, text: (res && res.text) || '', chunks: (res && res.chunks) || [] });
        } catch (err) {
          self.postMessage({ type: 'result', id: msg.id, error: String(err && err.message ? err.message : err) });
        }
      }
    };
  }

  // Runs inside the iframe. Its only job is to build the worker, because a Worker
  // created straight from the page is refused by the page's CSP.
  function bridgeMain() {
    var started = false;
    function tell(what, detail) {
      try { parent.postMessage({ type: 'bridge-' + what, detail: detail === undefined ? null : detail }, '*'); } catch (e) { }
    }
    tell('alive');
    window.onmessage = function (ev) {
      var msg = ev.data || {};
      var port = ev.ports && ev.ports[0];
      tell('message', { hasPort: !!port, hasSource: !!(msg && msg.workerSource) });
      if (!port || !msg.workerSource) return;
      port.start && port.start();
      if (started) { port.postMessage({ type: 'bridge-duplicate' }); return; }
      started = true;
      var url = null;
      try {
        url = URL.createObjectURL(new Blob([msg.workerSource], { type: 'text/javascript' }));
        var worker = new Worker(url);
        URL.revokeObjectURL(url);
        worker.onmessage = function (e) { port.postMessage(e.data); };
        worker.onerror = function (e) {
          port.postMessage({ type: 'init-error', error: 'worker blocked: ' + ((e && e.message) || 'no message') });
        };
        port.onmessage = function (e) { worker.postMessage(e.data); };
        tell('worker-created');
        port.postMessage({ type: 'bridge-ready' });
      } catch (err) {
        tell('worker-error', String(err));
        port.postMessage({ type: 'init-error', error: 'worker could not be created: ' + String(err) });
      }
    };
  }

  /* ------------------------------------------------------------------ *
   * Off-thread inference
   *
   * One Whisper call has a floor of roughly 1.5 s: the mel spectrogram is always
   * padded to 30 s, so the encoder cost is fixed however short the clip is. On the
   * main thread that is a visible freeze per censored word, so the work is pushed to
   * a Worker whenever one can be created.
   *
   * A Worker built from the page is refused by youtube.com's CSP, so the worker is
   * built inside a `data:` iframe injected with GM_addElement instead - that element
   * is added privileged, and a `data:` document does not carry the page's policy.
   * Plain DOM APIs do not get this; without GM_addElement the iframe inherits the CSP
   * and the whole path fails, in which case inference stays on the main thread.
   * ------------------------------------------------------------------ */

  const workerHost = {
    port: null,
    iframe: null,
    state: 'idle',     // idle | starting | ready | failed
    error: null,
    mode: 'main',      // where inference actually runs
    seq: 0,
    waiting: {},
    startPromise: null,

    start: function (model, classicSource) {
      if (this.startPromise) return this.startPromise;
      const host = this;
      this.state = 'starting';
      this.debug = { attempts: 0, alive: false, bridge: [] };
      this.handed = false;
      this.startPromise = new Promise(function (resolve, reject) {
        host.resolveStart = resolve;
        host.rejectStart = reject;
      });
      this.onAlive = function (ev) {
        const d = ev.data;
        if (!d || typeof d.type !== 'string' || d.type.indexOf('bridge-') !== 0) return;
        host.debug.bridge.push(d.type + (d.detail ? ' ' + JSON.stringify(d.detail).slice(0, 70) : ''));
        if (host.debug.bridge.length > 20) host.debug.bridge.shift();
        if (d.type === 'bridge-alive') host.debug.alive = true;
      };
      try { W.addEventListener('message', this.onAlive); } catch (e) {}

      const inner = '<!doctype html><title></title><scr' + 'ipt>(' + bridgeMain.toString() + ')();</scr' + 'ipt>';
      const src = 'data:text/html;charset=utf-8,' + encodeURIComponent(inner);
      const style = 'position:absolute!important;top:0!important;left:0!important;width:0!important;' +
                    'height:0!important;border:0!important;opacity:0!important;pointer-events:none!important';

      let iframe = null;
      try {
        if (typeof GM_addElement === 'function') {
          iframe = GM_addElement(document.documentElement, 'iframe', { style: style, src: src });
        }
      } catch (e) { iframe = null; }
      if (!iframe) {
        this.fail('GM_addElement is unavailable, so a worker cannot be created under this page policy');
        return this.startPromise;
      }
      this.iframe = iframe;

      const channel = new MessageChannel();
      this.port = channel.port1;
      this.port2 = channel.port2;
      this.port.onmessage = function (ev) { host.onMessage(ev.data); };
      if (this.port.start) this.port.start();

      // A MessagePort can only be transferred once, so the handshake must not be
      // retried with the same channel - the second attempt throws "Port at index 0 is
      // already neutered". The iframe announces itself when its script has run, and
      // only then is the port handed over.
      let waited = 0;
      const waitAlive = function () {
        if (host.state !== 'starting') return;
        if (host.debug.alive) { host.handshakeOnce(); return; }
        waited += 100;
        if (waited > 6000) { host.fail('the proxy iframe never came up'); return; }
        setTimeout(waitAlive, 100);
      };
      setTimeout(waitAlive, 100);

      this.pendingInit = {
        type: 'init', classicSource: classicSource,
        wasmBase: TRANSFORMERS_BASE, model: model,
      };
      return this.startPromise;
    },

    handshakeOnce: function () {
      if (this.handed) return;
      this.handed = true;
      const host = this;
      try {
        this.iframe.contentWindow.postMessage(
          { workerSource: '(' + workerMain.toString() + ')();' }, '*', [this.port2]
        );
      } catch (e) {
        this.fail('handshake failed: ' + (e && e.message ? e.message : e));
        return;
      }
      // No retry with this port; if the bridge stays silent, give up on the worker.
      // The allowance is generous because the worker downloads the bundle and the
      // model before it can report ready, and that is the slow part.
      setTimeout(function () {
        if (host.state === 'starting') host.fail('the proxy iframe never answered');
      }, 240000);
    },

    fail: function (reason) {
      this.state = 'failed';
      this.error = reason;
      this.mode = 'main';
      for (const id in this.waiting) {
        this.waiting[id].reject(new Error(reason));
        delete this.waiting[id];
      }
      if (this.rejectStart) { this.rejectStart(new Error(reason)); this.rejectStart = null; }
      if (this.iframe) { try { this.iframe.remove(); } catch (e) {} this.iframe = null; }
      updatePill();
    },

    onMessage: function (msg) {
      if (!msg) return;
      this.debug.port = this.debug.port || [];
      this.debug.port.push(msg.type + (msg.step ? ':' + msg.step : '') + (msg.error ? '!' + String(msg.error).slice(0, 60) : ''));
      if (this.debug.port.length > 40) this.debug.port.shift();
      if (msg.type === 'progress') return;
      if (msg.type === 'bridge-ready') {
        // The worker exists now; hand it the model configuration.
        try { this.port.postMessage(this.pendingInit); } catch (e) { this.fail(String(e)); }
        return;
      }
      if (msg.type === 'bridge-duplicate') return;
      if (msg.type === 'ready') {
        this.state = 'ready';
        this.mode = 'worker';
        this.error = null;
        if (this.resolveStart) { this.resolveStart('worker'); this.resolveStart = null; }
        updatePill();
        return;
      }
      if (msg.type === 'init-error') { this.fail(msg.error || 'worker init failed'); return; }
      if (msg.type === 'result') {
        const waiter = this.waiting[msg.id];
        if (!waiter) return;
        delete this.waiting[msg.id];
        if (msg.error) waiter.reject(new Error(msg.error));
        else waiter.resolve({ text: msg.text, chunks: msg.chunks });
      }
    },

    transcribe: function (pcm) {
      const host = this;
      const id = ++this.seq;
      return new Promise(function (resolve, reject) {
        host.waiting[id] = { resolve: resolve, reject: reject };
        const copy = new Float32Array(pcm);
        try {
          host.port.postMessage({ type: 'transcribe', id: id, pcm: copy.buffer }, [copy.buffer]);
        } catch (e) {
          delete host.waiting[id];
          reject(e);
        }
      });
    },
  };

  const asr = {
    pipe: null,
    pending: null,
    warmed: false,

    // Brings up whichever backend will do the work. Resolves to the main-thread
    // pipeline, or to null when the worker is the one that will run inference - it
    // must never resolve to anything the caller might try to call as a pipeline.
    ensure: function () {
      if (workerHost.state === 'ready') return Promise.resolve(null);
      if (this.pipe) return Promise.resolve(this.pipe);
      if (state.asrError) return Promise.resolve(null);   // do not retry a failed load
      if (!this.pending) {
        const self = this;
        this.pending = this.load().then(function (p) { self.pending = null; return p; },
                                        function (e) { self.pending = null; throw e; });
      }
      return this.pending;
    },

    load: async function () {
      state.asrLoading = true;
      state.asrError = null;
      updatePill();
      const model = (state.lang && state.lang.indexOf('en') === 0) ? config.modelEn : config.modelOther;

      // The worker is the only way to keep the page responsive, so it is tried first.
      // Its failure is immediate when the browser refuses, which is the common case.
      // The bundle is fetched and converted here so the worker only has to evaluate it.
      try {
        const classicSource = await fetchTransformersSource();
        await workerHost.start(model, classicSource);
      } catch (e) {
        // Fall through to running on the main thread.
      }
      if (workerHost.state === 'ready') {
        state.asrReady = true;
        state.asrLoading = false;
        updatePill();
        return null;
      }

      let pipe = null;
      try {
        const T = await loadTransformers();
        pipe = await T.pipeline('automatic-speech-recognition', model);
      } catch (e) {
        state.asrError = String(e && e.message ? e.message : e);
        state.asrLoading = false;
        updatePill();
        return null;
      }
      this.pipe = pipe;
      state.asrReady = true;
      state.asrLoading = false;
      updatePill();
      await this.warmUp();
      return pipe;
    },

    // The first call builds the ONNX session, grows the wasm heap and fills the
    // kernels' caches. Paying that while the video plays costs a visible hitch on
    // whichever censored cue happens to be first, so it is paid up front instead.
    warmUp: async function () {
      if (this.warmed || !this.pipe) return;
      this.warmed = true;
      state.asrWarming = true;
      updatePill();
      try {
        await this.pipe(new Float32Array(16000 * 5), { return_timestamps: 'word' });
      } catch (e) {
        // Warm-up is best effort; a failure here is not a reason to give up on ASR.
      } finally {
        state.asrWarming = false;
        updatePill();
      }
    },

    // One pipeline call. Returns the raw result; errors propagate so the caller can
    // tell "failed" from "found nothing", which decide different things about a job.
    transcribe: async function (audio) {
      // The backend is decided *after* ensure() resolves. Checking the worker first and
      // then calling ensure() leaves a window in which the worker comes up in between,
      // and ensure() would hand back a marker instead of a pipeline - which is exactly
      // how "pipe is not a function" happened.
      const pipe = await this.ensure();

      if (workerHost.state === 'ready') {
        const res = await workerHost.transcribe(audio.pcm);
        state.lastAsr = {
          where: 'worker',
          text: (res && res.text || '').slice(0, 200),
          chunks: (res && res.chunks) ? res.chunks.length : null,
          secs: +(audio.pcm.length / 16000).toFixed(2),
          audioStart: +audio.startSec.toFixed(2),
        };
        return res;
      }

      if (typeof pipe !== 'function') return null;
      // Do NOT pass `language` here. In @xenova/transformers 2.17.2, combining
      // `language` with `return_timestamps` makes the pipeline return an empty
      // transcript. The English model is already language-locked, and the
      // multilingual fallback auto-detects.
      const res = await pipe(audio.pcm, { return_timestamps: 'word' });
      let peak = 0;
      for (let i = 0; i < audio.pcm.length; i += 13) {
        const a = Math.abs(audio.pcm[i]);
        if (a > peak) peak = a;
      }
      state.lastAsr = {
        where: 'main',
        text: (res && res.text || '').slice(0, 200),
        chunks: (res && res.chunks) ? res.chunks.length : null,
        secs: +(audio.pcm.length / 16000).toFixed(2),
        peak: +peak.toFixed(3),
        pieces: audio.pieces,
        audioStart: +audio.startSec.toFixed(2),
        audioEnd: (audio.endSec === undefined) ? null : +audio.endSec.toFixed(2),
      };
      return res;
    },

    // Last resort when the transcript cannot be lined up with the caption's words:
    // take the word whose onset sits nearest the censored word's start.
    run: async function (audio, targetFromSec, targetToSec) {
      const res = await this.transcribe(audio);
      const chunks = res && res.chunks;
      if (!Array.isArray(chunks) || !chunks.length) return null;
      return pickByOnset(chunks, audio, targetFromSec, targetToSec);
    },
  };

  // The word whose onset sits nearest where the censored word starts. Used when the
  // transcript cannot be aligned against the caption, and as the tie-break inside it.
  function pickByOnset(chunks, audio, targetFromSec, targetToSec) {
    const onset = targetFromSec - audio.startSec;
    const spanFrom = onset;
    const spanTo = targetToSec - audio.startSec;
    let best = null;
    for (const c of chunks) {
      if (!c.timestamp || c.timestamp[0] === null) continue;
      const s = c.timestamp[0];
      const e = c.timestamp[1];
      const d = Math.abs(s - onset);
      const overlap = (e === null) ? 0 : Math.max(0, Math.min(e, spanTo) - Math.max(s, spanFrom));
      if (!best || d < best.d - 0.02 || (Math.abs(d - best.d) <= 0.02 && overlap > best.overlap)) {
        best = { d: d, overlap: overlap, text: c.text };
      }
    }
    // Nothing close enough to plausibly be the same word: leave the guess alone.
    if (!best || best.d > 0.6) return null;
    const word = cleanWord(best.text);
    return word || null;
  }

  // Line the transcript up against the caption's own word sequence for this cue.
  //
  // Matching by time does not work: YouTube's per-word tOffsetMs is only approximate
  // and was measured 0.4 s out on a real cue, which is wider than the gap between
  // words - that is how "slow as [ __ ]" came back as "as". The words either side of
  // the gap are known exactly, so the censored word is whatever Whisper puts at that
  // position. Alignment also resolves every censored word in the cue from one pass.
  function alignCue(cue, chunks) {
    const seq = cue.seq;
    if (!seq || !seq.length || !chunks.length) return null;
    const known = [];
    seq.forEach(function (item, i) {
      if (!item.gap && item.word) known.push({ seqIndex: i, word: item.word });
    });
    if (!known.length) return null;

    const said = chunks.map(function (c) { return stripWord(c.text); });
    let best = null;
    for (let j = 0; j < said.length; j++) {
      if (!said[j]) continue;
      for (let k = 0; k < known.length; k++) {
        if (said[j] !== known[k].word) continue;
        let back = 0;
        while (j - back - 1 >= 0 && k - back - 1 >= 0 &&
               said[j - back - 1] && said[j - back - 1] === known[k - back - 1].word) back++;
        let fwd = 0;
        while (j + fwd + 1 < said.length && k + fwd + 1 < known.length &&
               said[j + fwd + 1] && said[j + fwd + 1] === known[k + fwd + 1].word) fwd++;
        const score = back + fwd + 1;
        if (!best || score > best.score) {
          best = { score: score, chunkIndex: j, seqIndex: known[k].seqIndex };
        }
      }
    }
    return best;
  }

  // Fill in every censored word of one cue from a single transcript.
  function applyCueWords(cue, jobs, chunks, audio) {
    if (!Array.isArray(chunks) || !chunks.length) return 0;
    const alignment = alignCue(cue, chunks);
    let filled = 0;
    for (const job of jobs) {
      let word = null;
      if (alignment && alignment.score >= 2) {
        const chunkIndex = alignment.chunkIndex + (job.seqIndex - alignment.seqIndex);
        if (chunkIndex >= 0 && chunkIndex < chunks.length) {
          word = cleanWord(chunks[chunkIndex].text);
        }
      }
      if (!word) word = pickByOnset(chunks, audio, job.targetFrom, job.targetTo);
      if (word) {
        setCorrection(job.cueIdx, job.segIdx, word, 'asr');
        filled++;
      }
    }
    return filled;
  }

  function cleanWord(s) {
    const trimmed = String(s).replace(WS_G, ' ').trim();
    if (!trimmed) return null;
    // Whisper brackets non-speech ("[Music]", "[laughing]") and also parenthesises it
    // ("(laughing)", "(applause)"). Neither is ever the word that was censored, and
    // stripping the brackets would otherwise turn "(laughing)" into "laughing".
    if (/[[\]{}()<>*_\/\\|]/.test(trimmed)) return null;
    // Whisper attaches punctuation to the word (" shit," / "fuck!" / "Fucking."),
    // so strip what wraps it before checking that a single word is left.
    const stripped = trimmed.replace(/^[^A-Za-z]+/, '').replace(/[^A-Za-z'\-]+$/, '');
    if (!stripped || /\s/.test(stripped)) return null;
    const m = /^[A-Za-z][A-Za-z'\-]*$/.exec(stripped);
    return m ? m[0] : null;
  }

  /* ------------------------------------------------------------------ *
   * Queue pump
   *
   * The job list is never consumed. Each pass picks the earliest censored word whose
   * audio is currently inside the ring buffer, transcribes it and marks it done.
   * Words whose audio has already rolled out are left untouched, so starting in the
   * middle of a video and later seeking back re-enables them instead of losing them.
   * ------------------------------------------------------------------ */

  function nextEligibleJob(coveredTo, coveredFrom) {
    for (let i = 0; i < state.queue.length; i++) {
      const job = state.queue[i];
      if (job.done) continue;
      if (job.targetTo > coveredTo) continue;    // the word has not been spoken yet
      if (job.targetTo < coveredFrom) continue;  // its audio has rolled out of the buffer
      return job;
    }
    return null;
  }

  // The first cue that still has an unanswered censored word and whose audio the
  // archive already holds. Grouping by cue means one transcription resolves every
  // censored word in it, instead of one pipeline call per word.
  function nextArchiveCue() {
    const seen = {};
    for (let i = 0; i < state.queue.length; i++) {
      const job = state.queue[i];
      if (job.done) continue;
      if (seen[job.cueIdx]) continue;
      seen[job.cueIdx] = true;
      const jobs = state.queue.filter(function (j) { return !j.done && j.cueIdx === job.cueIdx; });
      let sliceFrom = Infinity, sliceTo = -Infinity, reqFrom = Infinity, reqTo = -Infinity;
      for (const j of jobs) {
        if (j.sliceFrom < sliceFrom) sliceFrom = j.sliceFrom;
        if (j.sliceTo > sliceTo) sliceTo = j.sliceTo;
        if (j.targetFrom - 0.15 < reqFrom) reqFrom = j.targetFrom - 0.15;
        if (j.targetTo + 0.25 > reqTo) reqTo = j.targetTo + 0.25;
      }
      if (!archive.coversRange(reqFrom, reqTo)) continue;
      return {
        cueIdx: job.cueIdx, jobs: jobs,
        sliceFrom: sliceFrom, sliceTo: sliceTo,
        requiredFrom: reqFrom, requiredTo: reqTo,
      };
    }
    return null;
  }

  async function runCue(group, audio) {
    if (!audio) {
      for (const job of group.jobs) {
        job.audioMisses++;
        if (job.audioMisses >= 4) job.done = true;
      }
      return false;
    }
    let peak = 0;
    for (let i = 0; i < audio.pcm.length; i += 11) {
      const a = Math.abs(audio.pcm[i]);
      if (a > peak) peak = a;
    }
    if (peak < 0.004) {
      // Nothing audible in the window. Whisper would only answer "[BLANK_AUDIO]",
      // so skip the inference and the main-thread hitch it costs.
      for (const job of group.jobs) job.done = true;
      return false;
    }

    state.pumping = true;
    state.busy = true;
    updatePill();
    try {
      const cue = state.cues[group.cueIdx];
      const res = await asr.transcribe(audio);
      const chunks = res && res.chunks;
      // One pass for the whole cue. Whether or not a word came back, do not run these
      // again: the audio is fixed, so a second pass would reach the same answer.
      applyCueWords(cue, group.jobs, chunks, audio);
      state.asrLastError = null;
      for (const job of group.jobs) job.done = true;
    } catch (e) {
      for (const job of group.jobs) {
        job.tries++;
        if (job.tries >= 3) job.done = true;
      }
      // A transcription error is not a load failure: state.asrError would make
      // ensure() refuse to ever try again, so one bad cue would disable ASR for the
      // rest of the video.
      state.asrLastError = String(e && e.message ? e.message : e);
    } finally {
      state.busy = false;
      state.pumping = false;
      updatePill();
    }
    return true;
  }

  async function runJob(job, audio) {
    if (!audio) {
      job.audioMisses++;
      if (job.audioMisses >= 4) job.done = true;
      return false;
    }
    let peak = 0;
    for (let i = 0; i < audio.pcm.length; i += 11) {
      const a = Math.abs(audio.pcm[i]);
      if (a > peak) peak = a;
    }
    if (peak < 0.004) {
      // Nothing audible in the window. Whisper would only answer "[BLANK_AUDIO]",
      // so skip the inference and the main-thread hitch it costs.
      job.done = true;
      return false;
    }

    state.pumping = true;
    state.busy = true;
    updatePill();
    try {
      const word = await asr.run(audio, job.targetFrom, job.targetTo);
      // Ran cleanly. Whether or not a word came back, do not run this cue again:
      // the audio is fixed, so a second pass would reach the same answer.
      job.done = true;
      if (word) setCorrection(job.cueIdx, job.segIdx, word, 'asr');
    } catch (e) {
      // A pipeline failure is worth one retry; the guess stays until then.
      job.tries++;
      state.asrError = String(e && e.message ? e.message : e);
      if (job.tries >= 2) job.done = true;
    } finally {
      state.busy = false;
      state.pumping = false;
      updatePill();
    }
    return true;
  }

  // The next word still ahead of the buffer, and how far ahead it is.
  function nextUpcomingJob() {
    let soonest = null;
    for (let i = 0; i < state.queue.length; i++) {
      const job = state.queue[i];
      if (job.done) continue;
      if (soonest === null || job.targetTo < soonest.targetTo) soonest = job;
    }
    return soonest;
  }

  function schedulePump(delay) {
    if (state.pumpTimer) return;
    state.pumpTimer = setTimeout(function () {
      state.pumpTimer = null;
      pumpQueue();
    }, delay);
  }

  async function pumpQueue() {
    if (state.pumping) return;
    if (state.busy) { state.pumpGate = 'transcribing'; return; }
    if (!config.enabled) { state.pumpGate = 'uncensor is off'; return; }
    if (!config.asr) { state.pumpGate = 'local speech recognition is off'; return; }
    if (!state.queue.length) { state.pumpGate = null; return; }

    const video = getVideo();
    if (!video) { state.pumpGate = 'no video element found'; return; }

    // Preferred path: the player has already buffered the audio, so the word can be
    // transcribed before the playhead - and therefore before the caption - gets there.
    if (archive.active()) {
      const ready = nextArchiveCue();
      if (ready) {
        const audio = archive.window(ready.sliceFrom, ready.sliceTo, ready.requiredFrom, ready.requiredTo);
        state.pumpGate = null;
        state.asrBlocked = null;
        await runCue(ready, audio);
        schedulePump(30);
        return;
      }
      if (archive.error) {
        state.asrBlocked = archive.error;
      } else if (hasPendingJobs()) {
        state.pumpGate = 'waiting for the player to buffer more audio';
      }
      scheduleDecode();
      schedulePump(250);
      return;
    }

    // Fallback: record the live output. This can only ever be behind the playhead, so
    // the correction lands late - it is here for when the media pipeline is not MSE.
    if (video.paused) { state.pumpGate = 'player is paused'; return; }

    const coveredTo = tap.coveredTo();
    if (!isFinite(coveredTo)) {
      state.pumpGate = 'waiting for audio';
      if (tap.attached && !tap.frames && Date.now() - tap.attachedAt > 5000) {
        state.asrBlocked = tap.reason || 'no audio captured from the player';
      }
      schedulePump(200);
      return;
    }
    const coveredFrom = coveredTo - RING_SECONDS;

    const job = nextEligibleJob(coveredTo, coveredFrom);
    if (!job) {
      const upcoming = nextUpcomingJob();
      state.pumpGate = upcoming
        ? 'next word in ' + Math.max(0, upcoming.targetTo - coveredTo).toFixed(1) + 's'
        : 'nothing left to transcribe';
      schedulePump(upcoming
        ? Math.max(30, Math.min(1000, (upcoming.targetTo - coveredTo) * 1000))
        : 300);
      return;
    }

    const cue = state.cues[job.cueIdx];
    const seg = cue && cue.segs[job.segIdx];
    if (!cue || !seg) { job.done = true; schedulePump(30); return; }

    // Four seconds of lead-in: whisper needs the surrounding words to settle on the
    // censored one, and it all has to sit behind the playhead. The required span is
    // slightly wider than the word itself so a drawn-out word is not cut at the edge.
    const audio = tap.slice16k(
      job.sliceFrom,
      job.sliceTo,
      job.targetFrom - 0.15,
      job.targetTo + 0.25
    );
    if (!audio) {
      state.pumpGate = 'waiting for audio';
      schedulePump(150);
      return;
    }
    state.pumpGate = null;
    state.asrBlocked = null;
    await runJob(job, audio);
    schedulePump(30);
  }

  function setCorrection(cueIdx, segIdx, word, source) {
    let map = state.corrections.get(cueIdx);
    if (!map) { map = new Map(); state.corrections.set(cueIdx, map); }
    const prev = map.get(segIdx);
    if (prev && prev.source === 'asr' && source === 'guess') return;
    map.set(segIdx, { word: word, source: source });
    if (source === 'asr') {
      const cue = state.cues[cueIdx];
      const seg = cue && cue.segs[segIdx];
      if (cue && seg) state.asrWords.set(wordKey(cue, seg), word);
    }
    applyToDom();
  }

  function cueWord(cueIdx, segIdx) {
    const map = state.corrections.get(cueIdx);
    const entry = map && map.get(segIdx);
    return entry ? entry.word : null;
  }

  /* ------------------------------------------------------------------ *
   * DOM patching
   * ------------------------------------------------------------------ */

  // YouTube keeps preview <video> elements for the sidebar and for hover previews, so
  // "the first video in the document" is not necessarily the player. Anchor on the
  // main player, and only fall back to a playing element.
  function getVideo() {
    const player = document.querySelector('.html5-video-player') || document.querySelector('#movie_player');
    if (player) {
      const inside = player.querySelector('video');
      if (inside) return inside;
    }
    const all = document.querySelectorAll('video');
    for (const v of all) if (!v.paused && v.readyState > 0) return v;
    return all.length ? all[0] : null;
  }

  function findCueForText(domText) {
    const norm = normalize(domText);
    if (!norm) return null;
    if (state.normIndex) {
      const exact = state.normIndex.get(norm);
      if (exact !== undefined) return exact;
    }
    // Not a whole cue: YouTube renders captions word by word, so this is usually a
    // fragment, and the fragment has to be located inside a cue.
    for (let i = 0; i < state.cues.length; i++) {
      const cue = state.cues[i];
      if (!cue.norm) continue;
      if (norm.indexOf(cue.norm) !== -1 || cue.norm.indexOf(norm) !== -1) return i;
    }
    return null;
  }

  function findCueAtTime(ms) {
    const cues = state.cues;
    if (!cues.length) return -1;
    // Cues are sorted by start time, so the only candidates are the few around the
    // last one that starts at or before this instant.
    let lo = 0, hi = cues.length - 1, best = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (cues[mid].startMs <= ms) { best = mid; lo = mid + 1; } else hi = mid - 1;
    }
    const from = Math.max(0, best - 8);
    const to = Math.min(cues.length - 1, best + 3);
    for (let i = from; i <= to; i++) {
      const cue = cues[i];
      if (ms >= cue.startMs - 400 && ms <= cue.startMs + Math.max(cue.durMs, 1200) + 400) return i;
    }
    return -1;
  }

  function fixText(domText, cueIdx) {
    if (!PLACEHOLDER.test(domText)) return null;
    const cue = state.cues[cueIdx];
    if (!cue) return null;

    // YouTube splits one cue across several .ytp-caption-segment elements, so this
    // element may not begin at the cue's first censored word. Counting the ones that
    // came before it is what keeps the second fragment from being answered with the
    // first fragment's word.
    let base = 0;
    const plain = normalize(domText);
    if (plain) {
      const at = cue.norm.indexOf(plain);
      if (at > 0) base = (cue.norm.slice(0, at).match(/\u0000/g) || []).length;
    }

    let n = 0;
    let changed = false;
    const out = domText.replace(PLACEHOLDER_G, function (whole) {
      const idx = base + n++;
      let seen = -1;
      for (let i = 0; i < cue.segs.length; i++) {
        if (!cue.segs[i].placeholder) continue;
        seen++;
        if (seen !== idx) continue;
        const word = cueWord(cueIdx, i);
        if (word) { changed = true; return word; }
        return whole;
      }
      return whole;
    });
    return changed ? out : null;
  }

  let domObserver = null;
  let applying = false;

  function installCaptionObserver() {
    if (domObserver) return;
    const target = document.querySelector('.ytp-caption-window-container')
      || document.querySelector('ytd-transcript-segment-list-renderer')
      || document.querySelector('.html5-video-player');
    if (!target) return;
    // YouTube rewrites the caption text on every word update, so patching only on a
    // timer leaves the raw placeholder visible for a frame. React to the mutation.
    domObserver = new MutationObserver(function () { applyToDom(); });
    domObserver.observe(target, { childList: true, subtree: true, characterData: true });
  }

  function applyToDom() {
    if (applying) return;
    if (!config.enabled || !state.cues.length) return;
    applying = true;
    try {
      patchCaptionSegments();
      patchTranscriptPanel();
    } finally {
      applying = false;
    }
  }

  function patchCaptionSegments() {
    const video = tap.attached || getVideo();
    const nowMs = video ? video.currentTime * 1000 : 0;

    const segs = document.querySelectorAll('.ytp-caption-segment');
    for (const el of segs) {
      const text = el.textContent;
      if (!text) continue;

      // Already patched and untouched since: a later (ASR) correction may have
      // changed the word, so re-derive from the text we originally saw. Without
      // this the guess would stay on screen because the element no longer holds a
      // placeholder to detect.
      const st = elState.get(el);
      if (st && text === st.written) {
        const again = fixText(st.raw, st.cueIdx);
        if (again && again !== text) {
          el.textContent = again;
          st.written = again;
        }
        continue;
      }

      if (!PLACEHOLDER.test(text)) continue;
      let cueIdx = findCueForText(text);
      if (cueIdx === null || cueIdx < 0) cueIdx = findCueAtTime(nowMs);
      if (cueIdx < 0) continue;
      const fixed = fixText(text, cueIdx);
      if (fixed && fixed !== text) {
        el.textContent = fixed;
        elState.set(el, { raw: text, cueIdx: cueIdx, written: fixed });
      }
    }
  }

  function patchTranscriptPanel() {
    // "Show transcript" panel: each row carries its own timestamp.
    const rows = document.querySelectorAll('ytd-transcript-segment-renderer');
    for (const row of rows) {
      const textEl = row.querySelector('.segment-text');
      if (!textEl) continue;
      const text = textEl.textContent;
      if (!text || !PLACEHOLDER.test(text)) continue;
      const tsEl = row.querySelector('.segment-timestamp');
      let cueIdx = -1;
      if (tsEl) {
        const parts = tsEl.textContent.trim().split(':').map(Number);
        if (parts.length && parts.every(function (n) { return !isNaN(n); })) {
          let secs = 0;
          for (const p of parts) secs = secs * 60 + p;
          cueIdx = findCueAtTime(secs * 1000);
        }
      }
      if (cueIdx < 0) cueIdx = findCueForText(text);
      if (cueIdx === null || cueIdx < 0) continue;
      const fixed = fixText(text, cueIdx);
      if (fixed && fixed !== text) textEl.textContent = fixed;
    }
  }

  /* ------------------------------------------------------------------ *
   * Player loop
   * ------------------------------------------------------------------ */

  let rafId = null;
  let lastTick = 0;

  function hasPendingJobs() {
    for (let i = 0; i < state.queue.length; i++) if (!state.queue[i].done) return true;
    return false;
  }

  // Start downloading and warming the model while there is still time, rather than
  // when the first censored word is already on screen.
  const PREWARM_AHEAD_MS = 180000;

  function maybePrewarm(video) {
    if (!config.enabled || !config.asr) return;
    if (asr.pipe || state.asrLoading || state.asrWarming || state.asrError) return;
    // With buffered audio in hand the model is on the critical path right now, no
    // matter where the playhead is.
    if (archive.active() && hasPendingJobs()) {
      asr.ensure().catch(function () {});
      return;
    }
    const nowMs = video.currentTime * 1000;
    let soonest = null;
    for (let i = 0; i < state.queue.length; i++) {
      const job = state.queue[i];
      if (job.done) continue;
      if (job.targetTo * 1000 < nowMs - 1000) continue;   // already behind the playhead
      if (soonest === null || job.targetTo < soonest) soonest = job.targetTo;
    }
    if (soonest === null) return;
    if (soonest * 1000 - nowMs > PREWARM_AHEAD_MS) return;
    asr.ensure().catch(function () {});
  }

  function tick() {
    rafId = requestAnimationFrame(tick);
    const now = Date.now();
    if (now - lastTick < 100) return;
    lastTick = now;

    const video = getVideo();
    if (!video) return;

    const videoId = currentVideoId();
    if (videoId && videoId !== state.videoId) resetVideoState(videoId);

    if (!config.enabled) return;

    if (!state.cues.length) {
      ensureCaptions();
      return;
    }

    if (config.asr && !archive.active()) {
      if (tap.attached !== video) {
        tap.attach(video);
        tap.fallbackTried = false;
      } else if (
        tap.frames === 0 && !video.paused && tap.mode === 'captureStream' &&
        !tap.fallbackTried && Date.now() - (tap.attachedAt || 0) > 5000
      ) {
        // captureStream produced nothing after several seconds of playback; take the
        // element's audio directly instead.
        tap.fallbackTried = true;
        tap.attach(video, true);
      }
      if (state.asrBlocked && tap.frames > 0 && tap.chunks.length) {
        state.asrBlocked = null;
        updatePill();
      }
    }

    installCaptionObserver();
    applyToDom();
    if (hasPendingJobs()) {
      // Prewarming is independent of audio capture: the model download should be
      // under way even while the tap is still being sorted out.
      maybePrewarm(video);
      if (!state.busy && !state.pumping && !state.asrBlocked) pumpQueue();
    }
  }

  function ensureCaptions() {
    if (!config.autoCaptions) return;
    const btn = document.querySelector('.ytp-subtitles-button');
    if (!btn) return;
    const on = btn.getAttribute('aria-pressed') === 'true';
    const now = Date.now();
    if (!on) {
      if (now - state.captionAttempt < 1500) return;
      state.captionAttempt = now;
      btn.click();
      return;
    }
    // Captions are on but nothing arrived: nudge the player a few times, then stop.
    if (state.captionRetries >= 3) return;
    if (now - state.captionAttempt < 4000) return;
    state.captionAttempt = now;
    state.captionRetries++;
    btn.click();
    setTimeout(function () {
      if (!state.cues.length) {
        const again = document.querySelector('.ytp-subtitles-button');
        if (again && again.getAttribute('aria-pressed') !== 'true') again.click();
      }
    }, 500);
  }

  function currentVideoId() {
    try {
      const url = new URL(location.href);
      if (url.pathname === '/watch') return url.searchParams.get('v');
      const m = /^\/(?:shorts|live|embed)\/([\w-]{6,})/.exec(url.pathname);
      if (m) return m[1];
    } catch (e) {}
    return null;
  }

  /* ------------------------------------------------------------------ *
   * UI
   * ------------------------------------------------------------------ */

  function updatePill() {
    const player = document.querySelector('.html5-video-player');
    if (!player) return;
    if (!state.pill || !state.pill.isConnected) {
      const pill = document.createElement('div');
      pill.className = 'ytasu-pill';
      pill.addEventListener('click', function (e) {
        e.stopPropagation();
        e.preventDefault();
        config.enabled = !config.enabled;
        store.write(config);
        updatePill();
        applyToDom();
        refreshMenu();
      });
      state.pill = pill;
      player.appendChild(pill);
    }
    const pill = state.pill;
    let text;
    let detail = '';
    if (!config.enabled) text = 'Uncensor: off';
    else if (!state.cues.length) {
      // A parse failure leaves no cues at all, which is otherwise indistinguishable
      // from captions simply not having arrived yet.
      if (state.parseError) { text = 'Uncensor: caption parse failed'; detail = state.parseError; }
      else text = 'Uncensor: waiting for captions';
    }
    else if (!config.asr) text = 'Uncensor: guess only';
    else if (state.asrError) { text = 'Uncensor: ASR failed'; detail = state.asrError; }
    else if (state.asrLastError) { text = 'Uncensor: ASR ready'; detail = 'last error: ' + state.asrLastError; }
    else if (state.asrBlocked) { text = 'Uncensor: no audio'; detail = state.asrBlocked; }
    else if (state.asrLoading) text = 'Uncensor: loading model…';
    else if (state.asrWarming) text = 'Uncensor: warming up…';
    else if (state.busy) text = 'Uncensor: transcribing…';
    else if (state.asrReady) text = 'Uncensor: ASR ready';
    else text = 'Uncensor: on';
    if (pill.textContent !== text) pill.textContent = text;
    pill.title = detail
      || ((config.asr && state.cues.length && state.pumpGate) ? state.pumpGate : '')
      || ('Inference: ' + (workerHost.mode === 'worker' ? 'worker thread' : 'main thread')
          + (workerHost.error ? ' (' + workerHost.error + ')' : ''));
    pill.classList.toggle('ytasu-off', !config.enabled);
    pill.classList.toggle('ytasu-error', !!(state.asrError || state.asrBlocked));
  }

  function installStyles() {
    const style = document.createElement('style');
    style.textContent = [
      '.ytasu-pill{position:absolute;left:12px;bottom:52px;z-index:60;',
      'font:11px/1.6 "YouTube Sans",Roboto,Arial,sans-serif;color:#eee;',
      'background:rgba(0,0,0,.62);border-radius:10px;padding:2px 9px;cursor:pointer;',
      'opacity:.55;transition:opacity .15s;user-select:none;pointer-events:auto;white-space:nowrap}',
      '.ytasu-pill:hover{opacity:1}',
      '.ytasu-pill.ytasu-off{color:#999}',
      '.ytasu-pill.ytasu-error{color:#ff9a9a}',
      '.html5-video-player.ytp-autohide .ytasu-pill{opacity:0;pointer-events:none}',
    ].join('');
    (document.head || document.documentElement).appendChild(style);
  }

  /* ------------------------------------------------------------------ *
   * Bootstrap
   * ------------------------------------------------------------------ */

  installCaptureHooks();

  function boot() {
    installStyles();
    updatePill();
    if (rafId === null) rafId = requestAnimationFrame(tick);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }

  // Debug handle: lets you inspect cue/correction state from the console.
  W.__ytasu = {
    state: state, config: config, applyToDom: applyToDom, tap: tap, store: store,
    asr: asr, setCorrection: setCorrection, pumpQueue: pumpQueue, updatePill: updatePill,
    parseTrack: parseTrack, analyze: analyze, maybePrewarm: maybePrewarm,
    archive: archive, alignCue: alignCue, applyCueWords: applyCueWords, runCue: runCue,
    nextArchiveCue: nextArchiveCue, workerHost: workerHost,
    fixText: fixText, findCueForText: findCueForText, findCueAtTime: findCueAtTime,
  };

  document.addEventListener('yt-navigate-finish', function () {
    const id = currentVideoId();
    if (!id || id === state.videoId) return;
    tap.detach();
    resetVideoState(id);
  });

  // Menu labels are fixed at registration time, so a toggle used to leave the old
  // "currently on" text in place until the page was reloaded - which reads as the
  // setting not having changed. Re-register after every change instead.
  const menuIds = [];

  function refreshMenu() {
    for (const id of menuIds) {
      try { if (typeof GM_unregisterMenuCommand === 'function') GM_unregisterMenuCommand(id); } catch (e) {}
    }
    menuIds.length = 0;

    const add = function (label, fn) {
      try {
        const id = GM_registerMenuCommand(label, fn);
        if (id !== undefined && id !== null) menuIds.push(id);
      } catch (e) {}
    };

    add((config.enabled ? '✓ ' : '✗ ') + 'Uncensor captions', function () {
      config.enabled = !config.enabled;
      store.write(config);
      updatePill();
      applyToDom();
      refreshMenu();
    });

    add((config.asr ? '✓ ' : '✗ ') + 'Recover the real word (local speech recognition)', function () {
      config.asr = !config.asr;
      store.write(config);
      updatePill();
      refreshMenu();
    });

    add((config.autoCaptions ? '✓ ' : '✗ ') + 'Turn captions on automatically', function () {
      config.autoCaptions = !config.autoCaptions;
      store.write(config);
      // Turning this off should also undo what it did, otherwise captions stay on from
      // an earlier click and the setting looks like it does nothing.
      if (!config.autoCaptions) {
        const btn = document.querySelector('.ytp-subtitles-button');
        if (btn && btn.getAttribute('aria-pressed') === 'true') btn.click();
      }
      refreshMenu();
    });
  }

  refreshMenu();
})();
