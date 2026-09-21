#!/usr/bin/env python3
"""Generate CC0-style procedural game audio (no external samples)."""
import math
import wave
import struct
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
SR = 22050


def write_wav(path, samples, sr=SR):
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    print(f"wrote {path} ({len(samples)/sr:.2f}s)")


def soft_env(n, attack=0.05, release=0.15, sr=SR):
    env = np.ones(n, dtype=np.float64)
    a = int(attack * sr)
    r = int(release * sr)
    if a > 0:
        env[:a] *= np.linspace(0, 1, a)
    if r > 0 and r < n:
        env[-r:] *= np.linspace(1, 0, r)
    return env


def nature_loop(seconds=12.0, sr=SR):
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(42)

    # Soft forest bed: filtered brownish noise
    noise = rng.standard_normal(n)
    # simple 1-pole lowpass (~800 Hz feel)
    bed = np.zeros(n)
    a = 0.015
    for i in range(1, n):
        bed[i] = bed[i - 1] + a * (noise[i] - bed[i - 1])
    bed *= 0.18

    # Distant water murmur: another filtered noise band modulated slowly
    water = rng.standard_normal(n)
    a2 = 0.04
    wfilt = np.zeros(n)
    for i in range(1, n):
        wfilt[i] = wfilt[i - 1] + a2 * (water[i] - wfilt[i - 1])
    water_mod = 0.55 + 0.45 * np.sin(2 * math.pi * 0.07 * t)
    bed += wfilt * 0.12 * water_mod

    # Soft drone tones (nature-y fifths)
    drone = (
        0.045 * np.sin(2 * math.pi * 110 * t)
        + 0.03 * np.sin(2 * math.pi * 165 * t)
        + 0.02 * np.sin(2 * math.pi * 220 * t + 0.3)
    )
    drone *= 0.7 + 0.3 * np.sin(2 * math.pi * 0.05 * t)

    # Occasional bird-like chirps
    chirps = np.zeros(n)
    chirp_times = [1.2, 2.8, 4.1, 5.6, 7.3, 8.9, 10.4]
    for ct in chirp_times:
        start = int(ct * sr)
        dur = int(0.18 * sr)
        if start + dur >= n:
            continue
        tt = np.arange(dur) / sr
        f0 = 1800 + (hash(str(ct)) % 700)
        freq = f0 + 600 * tt / 0.18  # upward slide
        amp = soft_env(dur, attack=0.01, release=0.08, sr=sr) * 0.07
        chirps[start : start + dur] += amp * np.sin(2 * math.pi * np.cumsum(freq) / sr)

    # Soft wind whoosh every few seconds
    wind = np.zeros(n)
    for wt in [0.5, 3.5, 6.8, 9.5]:
        start = int(wt * sr)
        dur = int(1.2 * sr)
        if start + dur >= n:
            continue
        wn = rng.standard_normal(dur)
        a3 = 0.08
        wf = np.zeros(dur)
        for i in range(1, dur):
            wf[i] = wf[i - 1] + a3 * (wn[i] - wf[i - 1])
        wind[start : start + dur] += wf * soft_env(dur, 0.3, 0.4, sr) * 0.06

    out = bed + drone + chirps + wind
    # Crossfade ends for seamless loop (~0.4s)
    xf = int(0.4 * sr)
    fade = np.linspace(0, 1, xf)
    out[:xf] = out[:xf] * fade + out[-xf:] * (1 - fade)
    out = out[:-xf]
    # gentle overall level
    peak = np.max(np.abs(out)) or 1.0
    out = out / peak * 0.55
    return out


def splash(sr=SR):
    dur = 0.65
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    # Noise burst (splash droplets)
    noise = rng.standard_normal(n)
    # band-ish via difference of smoothings
    a_hi, a_lo = 0.25, 0.05
    hi = np.zeros(n)
    lo = np.zeros(n)
    for i in range(1, n):
        hi[i] = hi[i - 1] + a_hi * (noise[i] - hi[i - 1])
        lo[i] = lo[i - 1] + a_lo * (noise[i] - lo[i - 1])
    band = hi - lo
    env_noise = np.exp(-t * 6.5) * (1 - np.exp(-t * 80))
    # Low whoosh / body
    whoosh = np.sin(2 * math.pi * (180 * np.exp(-t * 4) + 60) * t) * np.exp(-t * 3.2)
    # Bubble-ish mid blips
    blip = 0.35 * np.sin(2 * math.pi * 420 * t) * np.exp(-t * 8) * (t < 0.25)
    out = band * env_noise * 0.85 + whoosh * 0.45 + blip
    peak = np.max(np.abs(out)) or 1.0
    return out / peak * 0.9


def chaching(sr=SR):
    dur = 0.55
    n = int(dur * sr)
    t = np.arange(n) / sr
    # Classic register: two bright metallic hits + short ding decay
    def tone(freq, start, amp=0.55, decay=9.0, harm=True):
        s = np.zeros(n)
        i0 = int(start * sr)
        tt = t[i0:] - start
        if len(tt) == 0:
            return s
        wave_ = np.sin(2 * math.pi * freq * tt)
        if harm:
            wave_ += 0.45 * np.sin(2 * math.pi * freq * 2.01 * tt)
            wave_ += 0.22 * np.sin(2 * math.pi * freq * 3.02 * tt)
            wave_ += 0.12 * np.sin(2 * math.pi * freq * 4.5 * tt)
        env = np.exp(-tt * decay) * (1 - np.exp(-tt * 200))
        s[i0:] = amp * wave_ * env
        return s

    out = (
        tone(1318.5, 0.0, amp=0.5, decay=10)  # E6
        + tone(1760.0, 0.07, amp=0.55, decay=8)  # A6
        + tone(2093.0, 0.14, amp=0.35, decay=6)  # C7 sparkle
    )
    # tiny mechanical click
    click_n = int(0.02 * sr)
    rng = np.random.default_rng(3)
    click = rng.standard_normal(click_n) * soft_env(click_n, 0.001, 0.015, sr) * 0.25
    out[:click_n] += click
    peak = np.max(np.abs(out)) or 1.0
    return out / peak * 0.85




def chomp(sr=SR):
    """Alligator jaw slam / crunch for Level 2 fail."""
    dur = 0.55
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(11)
    thud = np.sin(2 * math.pi * (90 * np.exp(-t * 8) + 40) * t) * np.exp(-t * 6)
    noise = rng.standard_normal(n)
    a_hi, a_lo = 0.35, 0.08
    hi = np.zeros(n)
    lo = np.zeros(n)
    for i in range(1, n):
        hi[i] = hi[i - 1] + a_hi * (noise[i] - hi[i - 1])
        lo[i] = lo[i - 1] + a_lo * (noise[i] - lo[i - 1])
    crunch = (hi - lo) * np.exp(-t * 9) * (1 - np.exp(-t * 120))
    click = np.zeros(n)
    c0 = int(0.12 * sr)
    cd = int(0.08 * sr)
    cn = rng.standard_normal(cd)
    click[c0 : c0 + cd] = cn * soft_env(cd, 0.002, 0.05, sr) * 0.55
    squelch = 0.25 * np.sin(2 * math.pi * (220 + 80 * np.sin(2 * math.pi * 18 * t)) * t) * np.exp(-t * 7)
    out = thud * 0.7 + crunch * 0.95 + click * 0.5 + squelch
    peak = np.max(np.abs(out)) or 1.0
    return out / peak * 0.92

if __name__ == "__main__":
    write_wav(OUT / "nature-loop.wav", nature_loop())
    write_wav(OUT / "splash.wav", splash())
    write_wav(OUT / "cha-ching.wav", chaching())
    write_wav(OUT / "chomp.wav", chomp())
