#!/usr/bin/env python
"""Все соответствия «метка источника → специалист» живут здесь.

14 классов = 12 роутера + Педиатр и Онколог. Таблицы сверены с реальными
данными (data/raw/en/SCHEMAS.md): предмет MedMCQA даёт 9 классов напрямую,
тема — ещё 5; MTSamples и DDXPlus идут роутеру, HCM размечаем словарями.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Specialty:
    name: str        # русский класс (совпадает с классами роутера)
    slug: str        # имя модели: med-spec-<slug>-3b
    names_en: tuple[str, ...]  # английские названия для системного промпта


SPECIALTIES: tuple[Specialty, ...] = (
    Specialty("Терапевт",        "therapist",        ("internist", "general internal medicine")),
    Specialty("Хирург",          "surgeon",          ("general surgeon", "surgery")),
    Specialty("Гинеколог",       "gynecologist",     ("gynecologist", "obstetrician-gynecologist")),
    Specialty("Педиатр",         "pediatrician",     ("pediatrician", "pediatrics")),
    Specialty("Уролог",          "urologist",        ("urologist", "nephrologist")),
    Specialty("Онколог",         "oncologist",       ("oncologist", "oncology")),
    Specialty("Невролог",        "neurologist",      ("neurologist", "neurology")),
    Specialty("Дерматолог",      "dermatologist",    ("dermatologist", "dermatology")),
    Specialty("Офтальмолог",     "ophthalmologist",  ("ophthalmologist", "ophthalmology")),
    Specialty("Отоларинголог",   "ent",              ("ENT doctor", "otolaryngologist")),
    Specialty("Стоматолог",      "dentist",          ("dentist", "dental")),
    Specialty("Проктолог",       "proctologist",     ("proctologist", "colorectal surgeon")),
    Specialty("Гастроэнтеролог", "gastroenterologist", ("gastroenterologist", "gastroenterology")),
    Specialty("Травматолог",     "traumatologist",   ("traumatologist", "orthopedic surgeon")),
)

BY_NAME: dict[str, Specialty] = {s.name: s for s in SPECIALTIES}
BY_SLUG: dict[str, Specialty] = {s.slug: s for s in SPECIALTIES}

# --- MedMCQA: предмет → класс (базовые науки не мапим, они уходят в отчёт) ---
SUBJECT_MAP: dict[str, str] = {
    "Medicine": "Терапевт",
    "Surgery": "Хирург",
    "Gynaecology & Obstetrics": "Гинеколог",
    "Pediatrics": "Педиатр",
    "Skin": "Дерматолог",
    "Ophthalmology": "Офтальмолог",
    "ENT": "Отоларинголог",
    "Dental": "Стоматолог",
    "Orthopaedics": "Травматолог",
}

# Тема сильнее предмета. Порядок: частное раньше общего, онко раньше дефолтов.
CLINICAL_SUBJECTS = set(SUBJECT_MAP) | {"Psychiatry", "Unknown"}

_TOPIC_RULES: list[tuple[re.Pattern, str, bool]] = [
    (re.compile(r"rectum|anal canal|anal cancer|colorect|hemorrhoid|haemorrhoid"
                r"|polyps?[^.]*colon|colon[^.]*polyps|large intestine,? rectum", re.I),
     "Проктолог", False),
    (re.compile(r"oncol|cancer|tumor|tumour|carcinom|chemother|neoplasm|leukemia"
                r"|leukaemia|lymphoma|melanoma|sarcoma|metastas", re.I),
     "Онколог", False),
    (re.compile(r"neurology|epilep|seizure|stroke|cerebrovascular|parkinson|dementia"
                r"|multiple sclerosis|migraine|neuro-?ophthalmolog|\bCNS\b|central nervous"
                r"|nervous system|neuromuscular|myasthenia|neuropath|encephal|meningit", re.I),
     "Невролог", True),
    (re.compile(r"urolog|nephro|kidney|urinary|genitourin|\brenal\b|prostate|bladder|ureter"
                r"|urethr", re.I),
     "Уролог", False),
    (re.compile(r"g\.?i\.?t|gastro|hepat|\bliver\b|pancrea|biliary|esophag|oesophag"
                r"|intestin|stomach|duoden|colic fistula|gall ?stone", re.I),
     "Гастроэнтеролог", False),
]
_GALL_RE = re.compile(r"gall", re.I)


def map_medmcqa(subject: str, topic: str) -> str | None:
    """subject_name + topic_name MedMCQA → специалист или None."""
    s = str(subject or "").strip()
    t = str(topic or "").strip()
    if t and t.lower() not in {"nan", "unknown", "all india exam", "miscellaneous", "misc.",
                               "general", "fmge 2019", "fmge 2018", "neet jan 2020"}:
        for rx, name, clinical_only in _TOPIC_RULES:
            if rx.search(t):
                if clinical_only and s not in CLINICAL_SUBJECTS:
                    continue
                if name == "Уролог" and _GALL_RE.search(t):  # «Gall Bladder» — не урология
                    continue
                return name
    return SUBJECT_MAP.get(s)


# --- MTSamples (роутер) -----------------------------------------------------------
MTSAMPLES_MAP: dict[str, str] = {
    "Surgery": "Хирург",
    "Cardiovascular / Pulmonary": "Терапевт",
    "Orthopedic": "Травматолог",
    "General Medicine": "Терапевт",
    "Gastroenterology": "Гастроэнтеролог",
    "Neurology": "Невролог",
    "Obstetrics / Gynecology": "Гинеколог",
    "Urology": "Уролог",
    "ENT - Otolaryngology": "Отоларинголог",
    "Neurosurgery": "Хирург",
    "Hematology - Oncology": "Онколог",
    "Ophthalmology": "Офтальмолог",
    "Nephrology": "Уролог",
    "Pediatrics - Neonatal": "Педиатр",
    "Dentistry": "Стоматолог",
    "Dermatology": "Дерматолог",
    "Cosmetic / Plastic Surgery": "Хирург",
    "Podiatry": "Травматолог",
    "Endocrinology": "Терапевт",
    "Rheumatology": "Терапевт",
    "Allergy / Immunology": "Терапевт",
    "Bariatrics": "Хирург",
}


# --- DDXPlus (роутер/e2e): 49 фактических патологий → специальность -----------------
# Сверено со списком release_conditions.json; порядок важен: неоплазмы раньше
# гастро (Pancreatic neoplasm — Онколог, а не Гастроэнтеролог).
DDX_DISEASE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"neoplasm|cancer|tumor|carcinom", re.I), "Онколог"),
    (re.compile(r"otitis|sinusit|rhinitis|pharyngit|croup|epiglottit|laryn?gospasm|\bear\b",
                re.I), "Отоларинголог"),
    (re.compile(r"cluster headache|guillain|myasthenia|dystonic|neuropath", re.I), "Невролог"),
    (re.compile(r"hernia|boerhaave", re.I), "Хирург"),
    (re.compile(r"fracture", re.I), "Травматолог"),
    (re.compile(r"gerd|pancreat|food poisoning|gastro|esophag|biliar|gallstone", re.I),
     "Гастроэнтеролог"),
]
DDX_DEFAULT = "Терапевт"  # кардио/пульмо/инфекции/анемия/SLE/отёк/паника и пр.


def map_ddx_pathology(pathology: str) -> str:
    s = str(pathology or "").strip()
    for rx, name in DDX_DISEASE_RULES:
        if rx.search(s):
            return name
    return DDX_DEFAULT


# --- HealthCareMagic: словарная псевдо-разметка -------------------------------------
# Взвешенные ключевые слова: (паттерн, вес). Специальность присваивается при
# счёте ≥ 3; используем только для добора недостающих специальностей.
PSEUDO_KEYWORDS: dict[str, list[tuple[re.Pattern, int]]] = {
    "Проктолог": [
        (re.compile(r"\brectal\b|\brectum\b|\banus\b|\banal\b", re.I), 3),
        (re.compile(r"hemorrhoid|haemorrhoid|piles", re.I), 4),
        (re.compile(r"blood in stool|rectal bleeding|anal fissure|anal pain", re.I), 4),
        (re.compile(r"\bcolon\b|bowel movement", re.I), 1),
    ],
    "Невролог": [
        (re.compile(r"seizure|epileps", re.I), 4),
        (re.compile(r"migraine|numbness|tingling|tremor|paralysis", re.I), 2),
        (re.compile(r"\bstroke\b|\bTIA\b|parkinson", re.I), 4),
        (re.compile(r"\bheadache\b", re.I), 1),
    ],
    "Уролог": [
        (re.compile(r"prostate|prostatitis|BPH", re.I), 4),
        (re.compile(r"urinat|burning urin|dysuria|urinary", re.I), 2),
        (re.compile(r"kidney stone|renal colic|bladder", re.I), 3),
        (re.compile(r"erection|erectile|penis", re.I), 3),
    ],
    "Гастроэнтеролог": [
        (re.compile(r"acid reflux|heartburn|GERD", re.I), 4),
        (re.compile(r"gastrit|ulcer|stomach pain|abdominal pain", re.I), 2),
        (re.compile(r"\bliver\b|hepatitis|cirrhosis|jaundice", re.I), 3),
        (re.compile(r"diarrhea|constipation|bloating|IBS", re.I), 2),
    ],
    "Онколог": [
        (re.compile(r"\bcancer\b|\btumor\b|\btumour\b|malignan", re.I), 4),
        (re.compile(r"chemother|oncolog|metastas", re.I), 4),
        (re.compile(r"\blump\b|\bnodule\b", re.I), 1),
    ],
    "Дерматолог": [
        (re.compile(r"\brash\b|\bhives\b|urticaria", re.I), 4),
        (re.compile(r"\bitch\w*\b", re.I), 2),
        (re.compile(r"acne|pimple|eczema|psoriasis|dermatit", re.I), 4),
        (re.compile(r"\bmole\b|\bwart\b|skin (lump|lesion|bump)", re.I), 3),
        (re.compile(r"\bskin\b|\bscalp\b", re.I), 1),
    ],
    "Травматолог": [
        (re.compile(r"fracture|broken (bone|arm|leg|wrist|rib|ankle)", re.I), 4),
        (re.compile(r"\bsprain\b|\bstrain\b|dislocat", re.I), 3),
        (re.compile(r"torn (ligament|meniscus|ACL)|ACL tear|\btendon\b", re.I), 4),
        (re.compile(r"joint (pain|swelling|injury)", re.I), 2),
        (re.compile(r"\bcast\b|plaster|orthoped", re.I), 2),
    ],
}
PSEUDO_MIN_SCORE = 3


def pseudo_label(text: str) -> str | None:
    """HealthCareMagic input → специалист или None (по словарю, счёт ≥ PSEUDO_MIN_SCORE)."""
    best, best_score = None, 0
    for name, rules in PSEUDO_KEYWORDS.items():
        score = 0
        for rx, w in rules:
            hits = len(rx.findall(text or ""))
            if hits:
                score += w * min(hits, 2)
        if score >= PSEUDO_MIN_SCORE and score > best_score:
            best, best_score = name, score
    return best


if __name__ == "__main__":
    print(f"Специалистов: {len(SPECIALTIES)}")
    checks = [
        ("Medicine", "G.I.T", "Гастроэнтеролог"),
        ("Surgery", "Urology", "Уролог"),
        ("Gynaecology & Obstetrics", "Gynaecological oncology", "Онколог"),
        ("Medicine", "Cerebrovascular accident", "Невролог"),
        ("Anatomy", "Neuroanatomy", None),            # базовый предмет — нейро-правило клиническое
        ("Surgery", "Rectum and anal canal", "Проктолог"),
        ("Physiology", "Renal physiology", "Уролог"),
        ("Medicine", "nan", "Терапевт"),
        ("Anatomy", "Upper limb", None),
        ("Surgery", "Gall Bladder & Bile Ducts", "Хирург"),
    ]
    ok = True
    for subj, top, expect in checks:
        got = map_medmcqa(subj, top)
        mark = "OK " if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"  [{mark}] MedMCQA ({subj!r}, {top!r}) -> {got} (ожидалось {expect})")
    print("mtsamples Urology ->", MTSAMPLES_MAP.get("Urology"))
    print("ddx 'URTI' ->", map_ddx_pathology("URTI"), "| ddx 'Psoriasis' ->", map_ddx_pathology("Psoriasis"))
    print("pseudo 'rectal bleeding and hemorrhoids' ->", pseudo_label("I have rectal bleeding and hemorrhoids"))
    raise SystemExit(0 if ok else 1)
