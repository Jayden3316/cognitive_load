"""Step 6b: second opinion from Qwen3-Omni: tagged verbatim transcript (hesitations, laughter, swearing,
non-English words, cut-offs). Needs Pillow + torchvision, so it runs in its own env:
    HF_HUB_CACHE=/projects/data/speechteam/venkata/.cache/hf_models \
    /projects/data/speechteam/venkata/envs/qwen_omni/bin/python scripts/06b_second_opinion_qwen_omni.py"""
import glob, json, sys
import soundfile as sf, soxr, torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
PROMPT = """Transcribe this recording EXACTLY as spoken, word for word. It is a person speaking for about 30 seconds on a topic (Just A Minute game). This is for a disfluency study, so do NOT clean anything up and do NOT censor anything.

Rules:
- Keep every repetition, restart and false start. Write a cut-off word with a hyphen: "th-", "sp- specific".
- Mark every filled pause (hesitation sound) as [HESITATION: uh], [HESITATION: um], [HESITATION: er], [HESITATION: ah], [HESITATION: hmm], using the sound actually made.
- Mark laughter as [LAUGHTER] (also chuckles, giggles, laughing while speaking: [LAUGHTER] before the words said while laughing).
- Write any swear word or profanity in full, exactly as said (never with asterisks), and mark it: [SWEAR: word]. This includes swearing in Hindi or any other Indian language (write it in Latin letters).
- Write any words not in English in Latin letters and mark them: [NON-ENGLISH: words].
- Mark other non-speech sounds: [BREATH], [COUGH], [SIGH], [CLICK], [BACKGROUND SPEECH: words if audible].
- If the recording starts or ends in the middle of a word or sentence, write [CUT OFF] there.
- Output only the transcript text, nothing else."""


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/second_opinion/qwen3_omni.json"
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    model.disable_talker()
    out = {}
    for w in sorted(glob.glob("data/wav/*.wav")):
        x, sr = sf.read(w, dtype="float32")
        if sr != 16000:
            x = soxr.resample(x, sr, 16000)
        conv = [{"role": "user", "content": [{"type": "audio", "audio": x}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audio=[x], return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=600, do_sample=False, return_audio=False,
                                 thinker_return_dict_in_generate=False)
        ids = ids[0] if isinstance(ids, tuple) else ids
        ans = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        out[w.split("/")[-1][:-4]] = ans
        print(w.split("/")[-1][:-4], "|", ans[:160].replace("\n", " "), flush=True)
        json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
