import codecs

def fix_mojibake(text: str) -> str:
    """Reverse UTF-8-interpreted-as-Latin-1 mojibake (e.g. â -> ")."""
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

def decode_escape_sequences(text: str) -> str:
    """
    Decode escape sequences like \\n and \\u2011 in text.
    This converts literal escape sequences to their actual characters.
    """
    if not text:
        return text
    # First decode Unicode escape sequences (like \u2011)
    try:
        # Use codecs to decode unicode escapes
        text = codecs.decode(text, 'unicode_escape')
    except (UnicodeDecodeError, ValueError):
        # If decoding fails, try a more lenient approach
        print(f"Error decoding escape sequences: {e}")
        pass
        
    # Handle newlines and other common escapes
    # text = text.encode().decode('unicode_escape')
    return text

if __name__ == "__main__":
    with open("/root/multiagent_human_worker/tau-bench/tau_bench/RedditPersona.txt", "r", encoding="utf-8") as f:
        persona = f.read()
    persona = fix_mojibake(persona)
    clean_persona = decode_escape_sequences(persona)
    # s = clean_persona
    # i = s.find("year")
    # if i > 0:
    #     c = s[i-1]  # character before "year"
    #     print(repr(c), ord(c))  # e.g. '\u2011' 8209
    print(clean_persona)
    with open("clean_persona.txt", "w", encoding="utf-8") as f:
        f.write(clean_persona)
    