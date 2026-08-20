"""Guardrail policies: what to block, what to allow, and what to say instead.

Patterns are written to fire on *intent plus object*, not on keywords alone. A
keyword list that trips on "kill" refuses "how do I kill a hung process", which is
a worse failure than it looks: users stop trusting the system and route around it.
So each rule pairs an action verb with a harmful object, and `_ALLOWLIST` rescues
the technical idioms that would otherwise collide.

Coverage is deliberately Hindi + English, matching the deployed corpus languages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    SELF_HARM = "self_harm"
    WEAPONS = "weapons_cbrn"
    MINOR_SAFETY = "minor_safety"
    ILLICIT = "illicit"
    MALWARE = "malware"
    VIOLENCE = "violence"
    INJECTION = "prompt_injection"


@dataclass(frozen=True)
class Rule:
    category: Category
    pattern: re.Pattern[str]
    message: str


_F = re.IGNORECASE | re.UNICODE

# Every harmful-intent rule needs the same vocabulary of *asking*, so it is
# written once here. The rules below used to spell "how to" inline, one rule at a
# time, which meant "how to make a pipe bomb" was blocked and "how do I make a
# pipe bomb" sailed past the safety guard — a behavioural sweep only caught it
# because the retrieval guard happened to refuse that query for an unrelated
# reason. A shared fragment makes a new phrasing close the gap in every rule at
# once instead of in whichever rule someone remembers to edit.
_ASK = (
    r"(?:how\s+(?:to|do\s+(?:i|you|we)|can\s+(?:i|you|we)|would\s+i|does\s+one)"
    r"|steps?\s+(?:to|for)|instructions?\s+(?:to|for|on)|guide\s+(?:to|for|on)"
    r"|tutorial\s+(?:on|for)|recipe\s+for|teach\s+me(?:\s+how)?(?:\s+to)?"
    r"|show\s+me(?:\s+how)?(?:\s+to)?|tell\s+me\s+how\s+to"
    r"|(?:best|easiest|quickest|safest)\s+way\s+to|i\s+(?:want|need)\s+to"
    r"|help\s+me(?:\s+to)?|what'?s?\s+the\s+(?:best\s+)?(?:way|process|method)\s+to"
    r"|where\s+(?:can|do)\s+i\s+(?:get|buy|find|order|obtain))"
)
# Construction verbs in the inflections queries actually use. Requiring a verb is
# what keeps "what is sarin" answerable while "how do I make sarin" is not — the
# object alone is a topic, the verb plus the object is a request for help.
_BUILD = (
    r"(?:mak(?:e|ing)|build(?:ing)?|construct(?:ing)?|assembl(?:e|ing)"
    r"|manufactur(?:e|ing)|synthesi[sz](?:e|ing)|creat(?:e|ing)|cook(?:ing)?"
    r"|brew(?:ing)?|produc(?:e|ing))"
)
# Determiners and adjectives that sit between the verb and its object.
_A = (
    r"(?:\s+(?:a|an|the|my|your|own|some|home\s?made|improvised|working|real"
    r"|functional|diy))*\s+"
)

CRISIS_MESSAGE = (
    "I can't help with this, and I don't want to just leave it there. If you're "
    "thinking about harming yourself, please talk to someone now — in India, call "
    "or text the Tele-MANAS helpline on 14416 (24/7), or KIRAN on 1800-599-0019. "
    "Outside India, call your local emergency number. You deserve support from a "
    "person, not a search pipeline."
)

UNSAFE_RULES: tuple[Rule, ...] = (
    Rule(
        Category.SELF_HARM,
        re.compile(
            rf"\b(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life"
            rf"|commit\s+suicide|hang\s+myself|overdose\s+(?:on|to\s+die)"
            rf"|painless\s+(?:way|method)s?\s+to\s+die"
            rf"|{_ASK}\s+(?:die|kill\s+myself|end\s+it\s+all|hang\s+myself))\b"
            r"|आत्महत्या|खुदकुशी",
            _F,
        ),
        CRISIS_MESSAGE,
    ),
    Rule(
        Category.WEAPONS,
        re.compile(
            # The asking phrase is optional: a bare imperative ("make a pipe
            # bomb") is the same request with the politeness removed.
            rf"\b(?:{_ASK}\s+)?{_BUILD}{_A}"
            r"(?:pipe\s*bomb|nail\s*bomb|car\s*bomb|letter\s*bomb|dirty\s*bomb"
            r"|suicide\s+(?:bomb|vest)|bombs?|explosives?|detonator|ied"
            r"|improvised\s+explosive(?:\s+device)?"
            r"|nerve\s+agent|sarin|vx\s+gas|ricin|anthrax|mustard\s+gas"
            r"|chemical\s+weapon|bio(?:logical)?\s+weapon"
            r"|nuclear\s+(?:device|weapon|bomb)"
            r"|napalm|thermite|tatp|dynamite|molotov(?:\s+cocktail)?|c-?4)\b"
            r"|\b(?:untraceable\s+(?:gun|firearm)|ghost\s+gun"
            r"|3d[\s-]?print(?:ed|ing)?\s+(?:gun|firearm|receiver)"
            r"|convert\s+.{0,24}?\s+to\s+full\s+auto)\b"
            # Objects with no benign construction reading need no verb at all:
            # "recipe for TATP" is a request whether or not it says "make".
            # Deliberately excludes bomb/explosive/anthrax/mustard gas, which do
            # appear in mining, medical and history questions.
            rf"|\b{_ASK}{_A}"
            r"(?:pipe\s*bomb|nail\s*bomb|dirty\s*bomb|tatp|napalm|thermite"
            r"|nerve\s+agent|sarin|vx\s+gas|ricin|molotov(?:\s+cocktail)?"
            r"|ied|improvised\s+explosive(?:\s+device)?)\b"
            r"|बम\s*(?:कैसे\s*)?बना",
            _F,
        ),
        "I can't help with building weapons or explosives.",
    ),
    Rule(
        Category.MINOR_SAFETY,
        re.compile(
            r"(child (porn|sexual)|csam|minor[s]? (nude|naked|sexual)|"
            r"sexual(ly)? (explicit )?(content|images?) (of|with) (a )?(child|minor|kid))",
            _F,
        ),
        "I can't help with this request.",
    ),
    Rule(
        Category.ILLICIT,
        re.compile(
            rf"\b(?:{_ASK}\s+)?{_BUILD}{_A}"
            r"(?:meth|methamphetamine|crystal\s+meth|fentanyl|heroin|lsd|mdma)\b"
            # These objects, unlike sarin, are ordinary subjects of study: money
            # laundering is a compliance industry, counterfeit detection is a bank
            # function, document fraud is a crime reporters write about. So mention
            # alone cannot be the trigger — an asking phrase is mandatory here, for
            # the same reason it is in VIOLENCE. Measured before this was required:
            # 5 of 8 real legal/compliance questions were refused as unsafe,
            # including "penalties for laundering money under indian law" and "how
            # do banks detect fake passport applications".
            #
            # Intervening modifiers are the norm in real phrasing ("launder drug
            # money", "launder cartel cash"), and a rigid `launder\s+money` misses
            # every one of them — the same adjacency assumption that let "kill a
            # hung process" slip past the allowlist.
            rf"|\b{_ASK}(?:\s+\w+){{0,3}}?\s+"
            r"(?:launder(?:ing)?(?:\s+\w+){0,2}?\s+(?:money|cash|funds|proceeds|profits)"
            r"|(?:drugs|cocaine|meth|heroin|mdma)\s+online"
            r"|fake\s+(?:passport|aadhaar|id\s+card|currency)"
            r"|counterfeit\s+(?:money|currency|notes))\b",
            _F,
        ),
        "I can't help with illegal activity.",
    ),
    Rule(
        Category.MALWARE,
        re.compile(
            rf"\b(?:(?:{_ASK}\s+)?(?:writ(?:e|ing)|cod(?:e|ing)|{_BUILD})"
            r"\s+(?:me\s+)?(?:a\s+|an\s+|my\s+own\s+)?"
            r"(?:ransomware|keylogger|spyware|virus|trojan|botnet|worm|rootkit)|"
            r"ddos (attack|script|tool)|steal (credit card|password|credential)s?|"
            r"sql injection payload for|bypass (2fa|two.factor|antivirus)|"
            r"hack(ing)?\s+(into\s+)?(someone'?s?\s+|somebody'?s?\s+|my\s+|his\s+|her\s+|"
            r"their\s+|a\s+)?(\w+\s+){0,2}?"
            r"(account|wi-?fi|password|phone|instagram|facebook|gmail|e-?mail|snapchat|"
            r"whatsapp|router|server|database|webcam))\b",
            _F,
        ),
        "I can't help with malware or breaking into systems.",
    ),
    Rule(
        Category.VIOLENCE,
        # `_ASK` stays mandatory here: the objects are ordinary words, and a bare
        # "attack someone" appears in legitimate questions about self-defence law.
        re.compile(
            rf"\b(?:{_ASK}\s+(?:kill|murder|poison|hurt|attack)"
            r"\s+(?:a\s+|my\s+|someone|somebody|people|him|her|them)"
            r"|untraceable\s+poison|get\s+away\s+with\s+murder)\b",
            _F,
        ),
        "I can't help with harming people.",
    ),
)

# Technical idioms and consumer products that share vocabulary with the rules
# above. "how to make a bath bomb" is the single most common benign query that
# collides with the weapons rule, so it is listed first.
#
# The `kill` clause allows up to two intervening words ("kill a *hung* process"):
# an allowlist that only matched the bare noun let "how do I kill a hung process
# in linux" fall through to the violence rule the moment that rule learned to
# recognise "how do I". Refusing a sysadmin question is the failure mode this
# whole file is arranged to avoid, so the rescue has to be as flexible as the
# rule it is rescuing from.
_ALLOWLIST = re.compile(
    r"\b(bath\s?bombs?|smoke\s?bomb|glitter\s?bomb|seed\s?bomb|stink\s?bomb|"
    r"bomb\s?pop|photo\s?bomb|f[\s-]?bomb|"
    r"kill(?:ing)?\s+(?:(?:a|the|my|this|that|all|any|every)\s+)?(?:\w+\s+){0,2}?"
    r"(?:process(?:es)?|task|job|thread|container|pod|port|service|daemon|session|"
    r"instance|query|connection|server|node|app|program|window|tab|shell|terminal|"
    r"pid|kernel|socket)|"
    r"kill -9|pkill|"
    r"taskkill|killall|kill signal|dead(lock|line)|bomb(ing)? (cyclone|calorimeter)|"
    r"virus (protection|scanner|definition)|antivirus software|drug (interaction|dosage|"
    r"trial|store|test)|attack (surface|vector) (analysis|review))\b",
    _F,
)


INJECTION_RULES: tuple[Rule, ...] = (
    Rule(
        Category.INJECTION,
        re.compile(
            r"(ignore (all |any )?(previous|prior|above) (instruction|prompt|rule)s?"
            r"|disregard (your|the) (instructions|system prompt|guidelines)"
            r"|reveal (your|the) (system prompt|instructions|hidden rules)"
            r"|print (your|the) (system prompt|instructions)"
            r"|you are (now|no longer) (a|an|DAN)|developer mode|jailbreak"
            r"|pretend (you have|to have) no (rules|restrictions)"
            r"|repeat everything (above|before))",
            _F,
        ),
        "That looks like an attempt to override my instructions, so I've stopped here.",
    ),
)

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("card", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("aadhaar", re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")),
)

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in the indexed passages that answers this, so I'd "
    "rather not guess. Try rephrasing, or ask about something covered by the corpus."
)
UNGROUNDED_MESSAGE = (
    "I found related passages but they don't actually support a confident answer, "
    "so I'm declining rather than inventing one. The sources are listed below."
)
MALFORMED_MESSAGE = "I couldn't read a question in that. Could you rephrase it?"


def is_allowlisted(text: str) -> bool:
    """True when the text matches a benign technical idiom."""
    return bool(_ALLOWLIST.search(text))
