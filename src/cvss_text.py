"""Find and strip CVSS scores/vectors that some CNAs paste into the CVE description itself,
and build the scorer tags ("[NVD]", "[CNA: wordfence]") that v2 models read before the description.

Training on them lets the model copy the answer; at inference a quoted v3.x vector can be returned
as is. About 5% of descriptions contain one (mostly Oracle, Concrete CMS, Atlassian, runZero).
"""
import re

METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

# Full or bare vectors, any version, optionally wrapped in parentheses: CVSS:3.1/AV:N/..., (AV:N/AC:L/...)
_VECTOR = re.compile(r"\(?\s*(?:CVSS:[234]\.\d/|\b(?=AV:[NALP]/AC:[LMH]/))(?:[A-Za-z]{1,3}:[A-Za-z]/?)+\s*\)?")
# Oracle's "(Confidentiality and Integrity impacts)" spells out C/I/A
_IMPACTS = re.compile(r"\(\s*(?:Confidentiality|Integrity|Availability)[^()]*impacts?\s*\)", re.I)
# "(3.0 Low)" right after a vector
_SCORE_SEV = re.compile(r"\(\s*\d{1,2}\.\d\s*(?:None|Low|Medium|High|Critical)\s*\)", re.I)
# "CVSS v3.1 Base Score 7.5", "CVSS Score of 8.5", "base score of 5.3", "adjusted to 9.8", "CVSS 7.1"
_SCORE = re.compile(r"\b((?:CVSS\s*(?:v?\d(?:\.\d)?)?\s*)?(?:base\s*)?score(?:\s+(?:of|is|to be|adjusted to|:))?|CVSS)"
                    r"\s*:?\s*\d{1,2}\.\d\b", re.I)
_LABEL = re.compile(r"\bCVSS\s*(?:v?\d(?:\.\d)?)?\s*Vector\s*(?:of)?\s*:?\s*", re.I)
_V3 = re.compile(r"(?:CVSS:3\.[01]/)?(AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH])\b")


def find_vector(text: str) -> str | None:
    """First complete CVSS v3.x base vector quoted in the text, normalised to 'CVSS:3.1/...'."""
    m = _V3.search(text)
    return f"CVSS:3.1/{m.group(1)}" if m else None


def strip_cvss(text: str) -> str:
    """Remove quoted vectors, scores and impact hints; keep the rest of the sentence."""
    t = _VECTOR.sub(" ", text)
    t = _IMPACTS.sub(" ", t)
    t = _SCORE_SEV.sub(" ", t)
    t = _SCORE.sub(lambda m: m.group(1), t)          # keep "CVSS score", drop the number
    t = _LABEL.sub(" ", t)
    t = re.sub(r"\s+([.,;:)])", r"\1", t)              # tidy " ." left behind
    t = re.sub(r"\(\s*\)|\.\s*\.", lambda m: "" if m.group(0).startswith("(") else ".", t)
    return " ".join(t.split())


# ADPs and some CNAs are identified by UUID in the feeds
KNOWN_SCORERS = {"134c704f-9b21-4f2e-91b3-4a467353bcc0": "cisa-adp"}


def scorer_name(scorer: str) -> str:
    if scorer in KNOWN_SCORERS:
        return KNOWN_SCORERS[scorer]
    if "@" in scorer:
        return scorer.split("@")[1].rsplit(".", 1)[0]          # security@wordfence.com -> wordfence
    return scorer[:8]


def tag(scorer: str | None) -> str:
    return "[NVD]" if scorer is None else f"[CNA: {scorer_name(scorer)}]"
