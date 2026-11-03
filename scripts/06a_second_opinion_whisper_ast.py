"""Step 6a: second opinions. Whisper large-v3 (plain text, word times) and the AST AudioSet tagger
(laughter, cough, sigh, ... per 1 s window, every 0.25 s). Run in the repo .venv on a GPU."""
import json, glob, sys, numpy as np, soundfile as sf, torch
from transformers import pipeline, AutoFeatureExtractor, ASTForAudioClassification
dev = "cuda:0"
wavs = sorted(glob.glob("data/wav/*.wav"))
# 1) Whisper large-v3, word timestamps
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device=dev)
out = {}
for w in wavs:
    x, sr = sf.read(w, dtype="float32")
    r = asr({"raw": x, "sampling_rate": sr}, return_timestamps="word", chunk_length_s=30, stride_length_s=5,
            generate_kwargs={"language": "en", "task": "transcribe"})
    out[w.split("/")[-1][:-4]] = {"text": r["text"], "words": [{"word": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]} for c in r["chunks"]]}
json.dump(out, open("data/second_opinion/whisper_large_v3.json", "w"), indent=1)
print("whisper done")
# 2) AST AudioSet tagging, 1.0 s windows every 0.25 s
fe = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
m = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(dev).eval()
labels = m.config.id2label
want = [i for i, l in labels.items() if l in ("Laughter", "Giggle", "Snicker", "Chuckle, chortle", "Belly laugh", "Baby laughter",
        "Cough", "Sigh", "Breathing", "Gasp", "Throat clearing", "Sniff", "Speech", "Silence", "Groan", "Grunt", "Whimper", "Crying, sobbing", "Shout", "Yell", "Screaming", "Whistling", "Humming", "Clapping")]
ev = {}
for w in wavs:
    x, sr = sf.read(w, dtype="float32")
    win, hop = int(sr * 1.0), int(sr * 0.25)
    starts = list(range(0, max(1, len(x) - win + 1), hop))
    frames = []
    for b in range(0, len(starts), 64):
        batch = [x[s:s + win] for s in starts[b:b + 64]]
        inp = fe(batch, sampling_rate=sr, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.sigmoid(m(**inp).logits).cpu().numpy()
        frames += [{"t": round(s / sr, 2), **{labels[i]: round(float(pp[i]), 3) for i in want}} for s, pp in zip(starts[b:b + 64], p)]
    ev[w.split("/")[-1][:-4]] = frames
json.dump(ev, open("data/second_opinion/ast_events.json", "w"))
print("ast done")
