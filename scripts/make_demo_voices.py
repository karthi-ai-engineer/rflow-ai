"""The demo workspace's call voices (demo/workspace/audio/): one WAV per line of demo/workspace/scenario.js, and
audio/lines.js with each file's length and loudness (20 a second), which the call's speaking rings follow.

A line without its WAV gets a placeholder said by one of Windows' own Japanese voices (Windows.Media.SpeechSynthesis:
Ayumi, Ichiro, Sayaka, Haruka; a voice and a pitch per person, so the people sound apart). Real recordings replace them:
drop a WAV in under the same name (any rate, mono or stereo, 16-bit) and run this again; it only measures files that
are there. --force makes the placeholders again. Nothing is played: the voices go straight into files.

  uv run --no-sync python scripts/make_demo_voices.py [--force]
"""
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "demo" / "workspace"
AUDIO = WORKSPACE / "audio"
FPS = 20
# Windows' Japanese voices: who sounds like whom (a name in the voice, and a pitch: 1.0 as it is)
VOICES = {"yui": ("Ayumi", 1.0), "kenji": ("Ichiro", 0.92), "haruto": ("Ichiro", 1.12)}
FALLBACK = ("Haruka", 1.0)

SPEAK = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
[void][Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
  $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [type]$type) {
  $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
$jobs = Get-Content -Raw -Encoding UTF8 $args[0] | ConvertFrom-Json
foreach ($job in $jobs) {
  $s = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
  $voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices |
    Where-Object { $_.DisplayName -like "*$($job.voice)*" } | Select-Object -First 1
  if ($voice) { $s.Voice = $voice }
  $s.Options.AudioPitch = [double]$job.pitch
  $s.Options.SpeakingRate = 1.0
  $stream = Await ($s.SynthesizeTextToStreamAsync($job.text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
  $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
  $size = [uint32]$stream.Size
  [void](Await ($reader.LoadAsync($size)) ([uint32]))
  $bytes = New-Object byte[] $size
  $reader.ReadBytes($bytes)
  [System.IO.File]::WriteAllBytes($job.out, $bytes)
  Write-Output "$($job.out) ($($voice.DisplayName))"
}
"""


def lines() -> list[dict]:
    """The call's lines from scenario.js: who, file and the Japanese."""
    text = (WORKSPACE / "scenario.js").read_text(encoding="utf-8")
    found = re.findall(r'\{\s*who:\s*"(\w+)",\s*file:\s*"([^"]+)",\s*ja:\s*"([^"]+)"', text)
    return [{"who": who, "file": file, "ja": ja} for who, file, ja in found]


def speak(missing: list[dict]) -> None:
    jobs = [{"text": line["ja"], "out": str(AUDIO / line["file"]), "voice": VOICES.get(line["who"], FALLBACK)[0],
             "pitch": VOICES.get(line["who"], FALLBACK)[1]} for line in missing]
    with tempfile.TemporaryDirectory() as tmp:
        script, data = Path(tmp) / "speak.ps1", Path(tmp) / "jobs.json"
        script.write_text(SPEAK, encoding="utf-8-sig")
        data.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), str(data)], check=True)


def measure(path: Path) -> dict:
    """The file's length, and its loudness 20 times a second (0..1, the loudest stretch 1), smoothed a little."""
    with wave.open(str(path)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path.name}: 16-bit WAV please")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    step = rate // FPS
    n = max(1, len(samples) // step)
    rms = np.sqrt(np.array([np.mean(samples[i * step:(i + 1) * step] ** 2) for i in range(n)]) + 1e-9)
    level = np.clip(rms / (np.percentile(rms, 95) + 1e-9), 0, 1)
    level = np.convolve(level, np.ones(3) / 3, mode="same")
    level[rms < 150] = 0  # quiet: the ring rests
    return {"seconds": round(len(samples) / rate, 2), "fps": FPS, "env": [round(float(v), 2) for v in level]}


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    every = lines()
    if not every:
        raise SystemExit("no call lines found in scenario.js")
    missing = [line for line in every if "--force" in sys.argv or not (AUDIO / line["file"]).exists()]
    if missing:
        speak(missing)
    info = {line["file"]: measure(AUDIO / line["file"]) for line in every if (AUDIO / line["file"]).exists()}
    (AUDIO / "lines.js").write_text("// Made by scripts/make_demo_voices.py: each call line's length and loudness.\n"
                                    f"window.LINES = {json.dumps(info)};\n", encoding="utf-8")
    for name, i in info.items():
        print(f"{name}: {i['seconds']} s")


if __name__ == "__main__":
    main()
