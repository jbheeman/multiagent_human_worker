import codecs

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
    text = text.encode().decode('unicode_escape')
    return text

if __name__ == "__main__":
    with open("/root/multiagent_human_worker/tau-bench/tau_bench/RedditPersona.txt", "r", encoding="utf-8") as f:
        persona = f.read()
    clean_persona = decode_escape_sequences(persona)
    print(clean_persona)
    