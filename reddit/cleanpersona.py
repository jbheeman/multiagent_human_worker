import codecs
import json
import codecs
import re

def _fix_mojibake_once_mixed(text: str) -> str:
    if not text:
        return text

    out = []
    buf = []

    def flush():
        nonlocal buf
        if not buf:
            return
        s = "".join(buf)

        # Only try to repair chunks that look like mojibake
        if any(m in s for m in ("â", "Ã", "ð", "ï")):
            for enc in ("cp1252", "latin-1"):
                try:
                    s2 = s.encode(enc).decode("utf-8")
                    s = s2
                    break
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass

        out.append(s)
        buf = []

    for ch in text:
        if ord(ch) <= 0xFF:   # “8-bit-ish” characters (where mojibake lives)
            buf.append(ch)
        else:                 # keep real Unicode (like U+2011) as-is
            flush()
            out.append(ch)

    flush()
    return "".join(out)

def fix_mojibake(text: str) -> str:
    # Run twice to catch double-encoding
    for _ in range(2):
        new = _fix_mojibake_once_mixed(text)
        if new == text:
            break
        text = new
    return text
_CONTRACTIONS = [
    (re.compile(r"\bI[\u2010-\u2015]m\b"), "I'm"),
    (re.compile(r"\bdon[\u2010-\u2015]t\b"), "don't"),
    (re.compile(r"\bcan[\u2010-\u2015]t\b"), "can't"),
    (re.compile(r"\bwon[\u2010-\u2015]t\b"), "won't"),
    (re.compile(r"\bit[\u2010-\u2015]s\b"), "it's"),
    (re.compile(r"\bI[\u2010-\u2015]ll\b"), "I'll"),
    (re.compile(r"\bI[\u2010-\u2015]d\b"), "I'd"),
    (re.compile(r"\bI[\u2010-\u2015]ve\b"), "I've"),
]

def normalize_contractions(text: str) -> str:
    for pat, rep in _CONTRACTIONS:
        text = pat.sub(rep, text)
    return text

def decode_escape_sequences_if_needed(text: str) -> str:
    # json.loads already decodes \n and \uXXXX normally.
    # Only do this if the string is *double-escaped* and still contains literal backslashes.
    if not text:
        return text
    if "\\u" in text or "\\n" in text or "\\t" in text:
        try:
            return codecs.decode(text, "unicode_escape")
        except Exception:
            return text
    return text

def _clean_persona(persona: str) -> str:
    persona = fix_mojibake(persona)
    persona = normalize_contractions(persona)
    persona = decode_escape_sequences_if_needed(persona)
    return persona

if __name__ == "__main__":
    path = "/home/pgen/personagen/multiagent_human_worker/reddit/pipeline_personas.jsonl"
    output_path = "/home/pgen/personagen/multiagent_human_worker/reddit/pipeline_personas_clean.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(output_path, "w", encoding="utf-8") as f:
        for row in lines:
            row = row.strip()
            if not row:
                continue
            data = json.loads(row)
            persona = data["persona"]
            yaml = data["yaml"]
            # schwartz_values = data["schwartz_values"]
            clean_persona = _clean_persona(persona)
            print(clean_persona[:200], "...")
            print(yaml)
            f.write(json.dumps({"user_id": data["user_id"], "persona": clean_persona, "yaml": yaml}, ensure_ascii=False) + "\n")