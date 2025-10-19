"""
Compact element details system for WebSurfer agent.
Reduces token usage while maintaining accuracy for element identification.
"""

import re
import mmh3
from dataclasses import dataclass
from typing import List, Dict, Any, Set, Optional

# One-time codebook (sent once per run)
ELEMENT_CODEBOOK = {
    "ed_v": 1,
    "codes": {
        "tag": {"a": 0, "button": 1, "input": 2, "select": 3, "textarea": 4, "div": 5, "span": 6},
        "role": {"link": 0, "button": 1, "textbox": 2, "combobox": 3, "option": 4, "text": 5}
    }
}

TAG_CODE = ELEMENT_CODEBOOK["codes"]["tag"]
ROLE_CODE = ELEMENT_CODEBOOK["codes"]["role"]

# Action words that indicate interactive elements
ACTION_WORDS = {
    "login", "sign in", "search", "submit", "next", "continue", "apply", 
    "advanced", "filter", "checkout", "add to cart", "buy", "click", 
    "download", "upload", "save", "cancel", "confirm", "select", "choose",
    "go", "enter", "find", "browse", "navigate", "access", "open", "view",
    "show", "hide", "expand", "collapse", "more", "less", "all", "none"
}

def norm_text(s: str, maxlen: int = 24) -> str:
    """Normalize text: lowercase, collapse whitespace, truncate."""
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s[:maxlen]

def short_hash(s: str) -> str:
    """Generate 3-character hex hash for collision detection."""
    return format(mmh3.hash(s or "", signed=False) & 0xFFF, "03x")

def salience_score(el: Dict[str, Any]) -> int:
    """Calculate salience score for element ranking."""
    score = 0
    
    # Base score by element type
    role = el.get("role", "")
    tag = el.get("tag", "")
    
    if role in ("button", "link", "textbox"):
        score += 3
    if tag in ("button", "input", "select"):
        score += 2
    if el.get("in_form", False):
        score += 2
    if el.get("focused", False):
        score += 2
    if el.get("cta", False):  # Call-to-action
        score += 2
    
    # Boost for action words
    text = norm_text(el.get("text", "") or el.get("aria_label", ""))
    if any(word in text for word in ACTION_WORDS):
        score += 3
    
    # Special boost for search input fields
    if tag == "input" and role == "textbox":
        if "search" in text or text in ["", "search", "search..."]:
            score += 10  # High priority for search fields
        else:
            score += 5   # Medium priority for other input fields
    
    # Boost for visible interactive elements
    if el.get("visible", True) and el.get("interactive", False):
        score += 1
        
    return score

@dataclass
class MarkerElement:
    """Represents a single interactive element with marker."""
    id: int
    tag: str
    role: str
    text: str
    aria_label: Optional[str] = None
    visible: bool = True
    interactive: bool = True
    in_form: bool = False
    focused: bool = False
    cta: bool = False

def build_compact_ed(elements: List[MarkerElement], k: int = 16) -> Dict[str, Any]:
    """Build compact element details for top-K salient markers."""
    # Filter to visible elements only
    visible = [e for e in elements if e.visible]
    
    # Convert to dict format for scoring
    el_dicts = []
    for e in visible:
        el_dicts.append({
            "id": e.id,
            "tag": e.tag,
            "role": e.role,
            "text": e.text,
            "aria_label": e.aria_label,
            "visible": e.visible,
            "interactive": e.interactive,
            "in_form": e.in_form,
            "focused": e.focused,
            "cta": e.cta
        })
    
    # Rank by salience and take top K
    ranked = sorted(el_dicts, key=salience_score, reverse=True)[:k]
    
    # Build compact tuples
    ed = []
    for el in ranked:
        txt = norm_text(el.get("text") or el.get("aria_label", ""))
        ed.append([
            el["id"],
            TAG_CODE.get(el["tag"], -1),
            ROLE_CODE.get(el["role"], -1),
            txt,
            short_hash(txt)
        ])
    
    return {"ED": ed}

def diff_ids(prev_ids: List[int], curr_ids: List[int]) -> Dict[str, List[int]]:
    """Calculate delta between previous and current marker IDs."""
    ps, cs = set(prev_ids), set(curr_ids)
    return {
        "added": sorted(cs - ps),
        "removed": sorted(ps - cs), 
        "same": sorted(ps & cs)
    }

def format_element_details(ed_data: Dict[str, Any], delta: Optional[Dict[str, List[int]]] = None) -> str:
    """Format compact element details for inclusion in prompt."""
    ed = ed_data.get("ED", [])
    
    if not ed:
        return "No interactive elements found."
    
    lines = ["ELEMENTS (use numbered markers in screenshot):"]
    
    # Add delta info if provided
    if delta:
        if delta["added"]:
            lines.append(f"NEW: {delta['added']}")
        if delta["removed"]:
            lines.append(f"REMOVED: {delta['removed']}")
    
    # Add compact element info with concise descriptions
    for item in ed:
        id_val, tag_code, role_code, text, hash_val = item
        tag_name = next((k for k, v in TAG_CODE.items() if v == tag_code), "?")
        role_name = next((k for k, v in ROLE_CODE.items() if v == role_code), "?")
        
        text_display = text if text else ""
        
        # Add concise hints for important elements
        hint = ""
        if tag_name == "input" and role_name == "textbox":
            if "search" in text_display.lower() or text_display.lower() in ["", "search", "search..."]:
                hint = " 🔍SEARCH"
            else:
                hint = " INPUT"
        elif tag_name == "button":
            hint = " BTN"
        elif tag_name == "select" and role_name == "combobox":
            hint = " DROPDOWN"
        elif tag_name == "a" and role_name == "link":
            hint = " LINK"
        elif tag_name == "textarea":
            hint = " TEXTAREA"
            
        # Format: ID: type hint, text
        if text_display:
            lines.append(f"{id_val}: {tag_name}{hint}, '{text_display[:30]}'")
        else:
            lines.append(f"{id_val}: {tag_name}{hint}")
    
    return "\n".join(lines)
