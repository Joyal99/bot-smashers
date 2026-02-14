#!/usr/bin/env python3
"""
Bot or Not Challenge — Bot Detection Pipeline
=============================================
A multi-tier scoring system that identifies bot accounts in social media datasets.

Each user is scored across multiple tiers of signals. Users whose total score
meets or exceeds the DETECTION_THRESHOLD are flagged as bots.

Scoring rationale (competition: +4 TP, -1 FN, -2 FP):
  - False positives are costly (2x a missed bot), so we require converging signals.
  - Tier 1 signals are so reliable they can flag independently.
  - Tier 2/3 signals require combinations to reach the threshold.

Generalization notes:
  Signals are designed to detect general bot BEHAVIORS, not specific accounts.
  - GENERAL signals (timing regularity, batch posting, engagement patterns,
    encoding artifacts, meta-text leaks) form the detection backbone.
  - SPECIFIC signals (T2d 'just' frequency, T3c 'fun fact', T3d known openers)
    target observed bot algorithms. They cannot cause FPs on unseen data (they
    simply won't fire), but may miss new bot types with different quirks.
  - T3d includes a GENERALIZED dynamic opener check (first-3/4-word prefix
    repeated ≥5 times, T2-gated) to catch new template bots.
  - Defensive rules (T1/T2 gating, same-sec corroboration, spam exemption)
    prevent false positives and are conservative by design.

"""

import json
import re
import sys
import statistics
import argparse
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

DETECTION_THRESHOLD = 3  # Minimum score to flag as bot (tuned on practice data)

BASE_DIR = Path(__file__).resolve().parent
DATASET_POSTS_USERS_30 = BASE_DIR / "dataset.posts&users.30.json"
DATASET_POSTS_USERS_31 = BASE_DIR / "dataset.posts&users.31.json"
DATASET_POSTS_USERS_32 = BASE_DIR / "dataset.posts&users.32.json"
DATASET_POSTS_USERS_33 = BASE_DIR / "dataset.posts&users.33.json"
DATASET_BOTS_30 = BASE_DIR / "dataset.bots.30.txt"
DATASET_BOTS_31 = BASE_DIR / "dataset.bots.31.txt"
DATASET_BOTS_32 = BASE_DIR / "dataset.bots.32.txt"
DATASET_BOTS_33 = BASE_DIR / "dataset.bots.33.txt"


def _resolve_dataset_alias(dataset_arg):
    """Resolve dataset aliases ('30'/'32') or return provided path as-is."""
    aliases = {
        "30": DATASET_POSTS_USERS_30,
        "31": DATASET_POSTS_USERS_31,
        "32": DATASET_POSTS_USERS_32,
        "33": DATASET_POSTS_USERS_33,
    }
    if dataset_arg in aliases:
        return str(aliases[dataset_arg])
    return dataset_arg


def _default_bots_for_dataset(dataset_path):
    """Pick default bots file when dataset is one of the known challenge files."""
    name = Path(dataset_path).name
    if name == "dataset.posts&users.30.json":
        return str(DATASET_BOTS_30)
    if name == "dataset.posts&users.31.json":
        return str(DATASET_BOTS_31)
    if name == "dataset.posts&users.32.json":
        return str(DATASET_BOTS_32)
    if name == "dataset.posts&users.33.json":
        return str(DATASET_BOTS_33)
    return None


def _dataset_id_from_path(dataset_path):
    """Infer challenge dataset id (30/32) from filename when possible."""
    name = Path(dataset_path).name
    if name == "dataset.posts&users.30.json":
        return "30"
    if name == "dataset.posts&users.31.json":
        return "31"
    if name == "dataset.posts&users.32.json":
        return "32"
    if name == "dataset.posts&users.33.json":
        return "33"
    return "custom"


def _output_path_for_dataset(base_output, dataset_path):
    """Create dataset-specific output file when processing multiple datasets."""
    dataset_id = _dataset_id_from_path(dataset_path)
    output = Path(base_output)
    if output.suffix:
        return str(output.with_name(f"{output.stem}.{dataset_id}{output.suffix}"))
    return str(output.with_name(f"{output.name}.{dataset_id}.txt"))


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def parse_timestamps(posts):
    """Parse ISO timestamps from posts into datetime objects."""
    times = []
    for p in sorted(posts, key=lambda x: x["created_at"]):
        t = p["created_at"].replace("Z", "+00:00")
        times.append(datetime.fromisoformat(t))
    return times


def compute_intervals(times):
    """Compute inter-tweet intervals in seconds."""
    return [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]


def compute_cv(intervals):
    """Compute coefficient of variation of intervals (std / mean)."""
    if len(intervals) < 3:
        return None  # Not enough data
    mean = statistics.mean(intervals)
    if mean == 0:
        return 0.0
    return statistics.stdev(intervals) / mean


# ============================================================================
# TIER 1 — NEAR-CERTAIN SIGNALS (10 points each)
# ============================================================================
# These signals have essentially zero false positive risk. Any one of them
# alone is enough to confidently flag a bot.

# --- Tier 1a: Meta-Text / Leaked LLM Prompts ---
# Some bots are generated by LLMs that leak their instructions into the output.
# E.g., "Here are some of my recent tweets:" or "Here's a revised version."
# Real humans never write tweets like this.

META_PHRASES = [
    "here are some of my recent tweets",
    "here are my recent tweets",
    "here are some of my latest tweets",
    "here are some recent rewrites",
    "here are the revised versions",
    "here are some modified versions",
    "here are some alternatives",
    "here are some rewrites",
    "here are some changes",
    "here's a slightly modified version",
    "here's a slightly altered version",
    "here's a minor revision",
    "here's a minor revised version",
    "here's a minor re-phrased version",
    "here's a minor change",
    "here's a lightly rephrased version",
    "here's a subtle rewrite",
    "here's the revised version",
    "here's a stat for you",  # Common bot copy pattern
    "revised version of the",
    "modified version of",
    "slightly modified version",
    "some of my recent tweets",
    "my recent tweets",
    "rewritten with minor changes",
    "content generation assistant",
    "you are a content generation assistant",
    "you are trained on data up to",
]

META_PHRASES_FR = [
    "voici mes tweets",
    "voici quelques tweets",
    "voici quelques-uns de mes tweets",
    "voici une version révisée",
    "voici une version modifiée",
    "en tant que modèle de langage",
    "je suis un modèle de langage",
    "assistant de génération de contenu",
    "vous êtes un assistant",
]



def tier1a_meta_text(texts, lang="en"):
    """
    TIER 1a: Leaked LLM prompt / meta-text detection.
    
    Scans all tweets for phrases that indicate the text was generated by an LLM
    that leaked its system prompt or instruction framing into the output.
    
    Returns: (score, count of meta-text matches)
    """
    phrases = META_PHRASES if lang != "fr" else (META_PHRASES + META_PHRASES_FR)
    count = 0
    for text in texts:
        text_lower = text.lower()
        for phrase in phrases:
            if phrase in text_lower:
                count += 1
                break  # One match per tweet is enough
    
    # 2+ matches = very confident (10 pts)
    # 1 match could be coincidence, give partial credit (2 pts)
    if count >= 2:
        return 10, count
    elif count == 1:
        return 2, count
    return 0, count


# --- Tier 1b: Encoding Artifacts / Garbled Text ---
# Some bot generation pipelines produce text with control characters or
# encoding corruption (e.g., \x00-\x1f bytes). Real scraped tweets from
# Twitter's API never contain these. Zero false positives observed.

GARBLED_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def tier1b_encoding_artifacts(texts):
    """
    TIER 1b: Encoding artifact detection.
    
    Checks for control characters in tweet text that indicate corrupted
    text generation pipeline output.
    
    Returns: (score, count of garbled tweets)
    """
    count = sum(1 for t in texts if GARBLED_PATTERN.search(t))
    
    if count >= 1:
        return 10, count
    return 0, count


# ============================================================================
# TIER 2 — STRONG SIGNALS (3-5 points each)
# ============================================================================
# These signals have low but non-zero false positive risk. They are strong
# enough to contribute significantly to the score, but typically need at
# least one other signal to reach the threshold.

# --- Tier 2a: Same-Second Tweet Bursts ---
# Bots often dump multiple tweets at the exact same timestamp (same second),
# which happens when a script generates and posts them in a batch. Real users
# occasionally have a few (thread posts, API imports), but bots have many more.

def tier2a_same_second_bursts(posts):
    """
    TIER 2a: Same-second tweet burst detection.
    
    Counts how many tweets share an exact timestamp (to the second).
    Bots that batch-post will have many; humans rarely exceed 2-3.
    
    Returns: (score, count of duplicate-second tweets)
    """
    time_counts = Counter(p["created_at"] for p in posts)
    same_second = sum(v - 1 for v in time_counts.values() if v > 1)
    
    if same_second >= 5:
        return 5, same_second
    elif same_second >= 3:
        return 3, same_second
    return 0, same_second


# --- Tier 2b: Posting Interval Regularity (CV) ---
# The coefficient of variation (std/mean) of inter-tweet time gaps.
# Humans post erratically (bursts then silence) → high CV (~1.8-2.0).
# Bots post more evenly → low CV (~1.0-1.3).
# This is the single strongest statistical signal across both practice datasets.

def tier2b_interval_regularity(times):
    """
    TIER 2b: Posting interval regularity via coefficient of variation.
    
    Low CV = suspiciously regular posting pattern.
    
    Post-count tiers:
      12-14 posts: only trigger for very low CV (≤ 0.90) → 4 pts
      ≥15 posts: full threshold range
        CV ≤ 0.8  → 5 pts
        CV ≤ 1.0  → 4 pts
        CV ≤ 1.15 → 3 pts
    
    Returns: (score, cv_value or None)
    """
    if len(times) < 10:
        return 0, None  # Not enough posts for reliable CV
    
    intervals = compute_intervals(times)
    cv = compute_cv(intervals)
    
    if cv is None:
        return 0, None
    
    if len(times) < 12:
        # Ultra-strict threshold for 10-11 posts — only very regular patterns
        if cv <= 1.10:
            return 2, cv
        return 0, cv
    
    if len(times) < 15:
        # Strict threshold for 12-14 posts
        if cv <= 0.90:
            return 4, cv
        return 0, cv
    
    # Standard thresholds for 15+ posts
    if cv <= 0.8:
        return 5, cv
    elif cv <= 1.0:
        return 4, cv
    elif cv <= 1.15:
        return 3, cv
    elif cv <= 1.20:
        return 2, cv
    return 0, cv


# --- Tier 2c: Template Pattern (Zero URL + Zero Hashtag) ---
# A specific class of bots generates tweets from fill-in-the-blank templates
# (e.g., "The ending of [MOVIE] still messes me up."). These bots produce
# pure text with no URLs or hashtags, and at decent volume. Real users who
# tweet 15+ times almost always include at least some URLs or hashtags.

def tier2c_template_pattern(texts, n_posts):
    """
    TIER 2c: Template bot detection via zero-URL + zero-hashtag pattern.
    
    Catches bots that generate text from templates without any links or tags.
    Requires ≥30 posts — at lower volumes, some humans naturally don't
    include URLs or hashtags, causing false positives.
    
    Returns: (score, is_template_pattern: bool)
    """
    url_rate = sum(1 for t in texts if "http" in t) / max(len(texts), 1)
    hashtag_rate = sum(t.count("#") for t in texts) / max(len(texts), 1)
    
    if url_rate == 0 and hashtag_rate == 0 and n_posts >= 30:
        return 5, True
    return 0, False


# --- Tier 2d: High 'just' Frequency (Quirky Anecdote Bots) ---
# A class of bots generates relatable anecdote tweets that heavily overuse
# the word "just" — "just walked into a glass door", "just waved at someone",
# "just accidentally liked a photo from 5 years ago". Humans use "just" at
# ~5-6% of tweets; these bots hit 35-60%. At ≥0.35 with ≥15 posts, this
# signal has ZERO false positives across both practice datasets.

VIENS_DE_PATTERN = re.compile(r"\b(j'?viens|viens|vient)\s+de\b", re.IGNORECASE)

def tier2d_just_frequency(texts, n_posts, lang="en"):
    """
    TIER 2d: Overuse of 'just' indicating quirky-anecdote bot pattern.
    
    Graduated scoring with tiered thresholds by post count:
      n ≥  8 and 'just' rate ≥ 0.40  → 4 pts (strong even at low volume)
      n ≥ 15 and 'just' rate ≥ 0.35  → 4 pts (standard threshold)
      n ≥ 15 and 'just' rate ≥ 0.28  → 2 pts (moderate, needs combo)
    
    Returns: (score, just_rate)
    """
    if n_posts < 8:
        return 0, 0.0
    
    if lang == "fr":
        hits = sum(1 for t in texts if VIENS_DE_PATTERN.search(t))
    else:
        hits = sum(1 for t in texts if re.search(r"\bjust\b", t.lower()))
    just_rate = hits / n_posts
    
    if n_posts >= 15 and just_rate >= 0.35:
        return 4, just_rate
    elif n_posts >= 8 and just_rate >= 0.40:
        return 4, just_rate
    elif n_posts >= 15 and just_rate >= 0.28:
        return 2, just_rate
    return 0, just_rate


# --- Tier 2e: Zero-Engagement Pattern (No URLs + No Mentions) ---
# Real Twitter users almost always interact — sharing links, replying to
# people, or mentioning accounts. Bots that generate synthetic content
# into the void have zero URLs AND zero @mentions. This is a broader
# version of the template check (Tier 2c) that catches bots which DO
# use hashtags but never engage with real content or people.

def tier2e_zero_engagement(texts, n_posts):
    """
    TIER 2e: Zero-engagement pattern — no URLs and no @mentions.
    
    Catches bots that generate content but never link to anything or
    interact with other accounts. Requires ≥15 posts.
      url_rate=0 AND mention_rate=0 → 2 pts
    
    Returns: (score, is_zero_engagement: bool)
    """
    if n_posts < 15:
        return 0, False
    
    url_rate = sum(1 for t in texts if "http" in t) / n_posts
    mention_rate = sum(1 for t in texts if "@" in t) / n_posts
    
    if url_rate == 0 and mention_rate == 0:
        return 2, True
    return 0, False


# ============================================================================
# TIER 3 — SUPPORTING SIGNALS (1-2 points each)
# ============================================================================
# Weak signals that add evidence when combined with other tiers.
# Never enough to flag a bot alone, but push borderline cases over the edge.

# --- Tier 3a: Elevated Hashtag Rate ---
# Bots average ~0.9 hashtags/tweet vs ~0.2 for humans.
# High hashtag density suggests automated content seeding.

def tier3a_hashtag_density(texts):
    """
    TIER 3a: Hashtag density analysis.
    
    Bots tend to use significantly more hashtags than humans.
      ≥1.0 hashtags/tweet → 2 pts
      ≥0.5 hashtags/tweet → 1 pt
    
    Returns: (score, hashtag_rate)
    """
    hashtag_rate = sum(t.count("#") for t in texts) / max(len(texts), 1)
    
    if hashtag_rate >= 1.0:
        return 2, hashtag_rate
    elif hashtag_rate >= 0.5:
        return 1, hashtag_rate
    return 0, hashtag_rate


# --- Tier 3b: Very Low URL Rate ---
# Real users share links frequently (~52% of tweets contain URLs).
# Bots that generate original text rarely include real URLs (~29%).
# Very low URL rate at decent volume is suspicious.

def tier3b_low_url_rate(texts, n_posts):
    """
    TIER 3b: Unusually low URL sharing rate.
    
    Most humans share links regularly. Bots generating synthetic content
    rarely include URLs. Requires ≥15 posts to be meaningful.
      URL rate ≤ 10% with ≥15 posts → 1 pt
    
    Returns: (score, url_rate)
    """
    url_rate = sum(1 for t in texts if "http" in t) / max(len(texts), 1)
    
    if url_rate <= 0.10 and n_posts >= 15:
        return 1, url_rate
    return 0, url_rate


# --- Tier 3c: 'Fun Fact' Pattern ---
# Quirky-anecdote bots frequently inject "fun fact:" into their tweets
# as a filler device. E.g., "walked into a glass door. fun fact: birds
# do it too." Zero humans in practice data use this phrase 2+ times.

FUN_FACT_PATTERN = re.compile(r'\bfun fact\b', re.IGNORECASE)
FUN_FACT_FR_PATTERN = re.compile(r"\ble saviez[- ]vous\b|\bfun fact\b", re.IGNORECASE)



def tier3c_fun_fact(texts, lang="en"):
    """
    TIER 3c: Repeated 'fun fact' usage.
    
    Quirky-anecdote bots use this phrase as a filler device. Humans
    almost never use it more than once.
      'fun fact' count ≥ 2 → 2 pts
    
    Returns: (score, fun_fact_count)
    """
    if lang == "fr":
        count = sum(1 for t in texts if FUN_FACT_FR_PATTERN.search(t))
    else:
        count = sum(1 for t in texts if FUN_FACT_PATTERN.search(t))
    
    if count >= 2:
        return 2, count
    return 0, count


# --- Tier 3d: Repetitive Tweet Opener ---
# Some bots start many tweets with the same phrase, revealing a template.
# E.g., "Remember when..." appearing as the opener in 15/36 tweets.
# Humans don't repeat opening phrases this aggressively.

REPETITIVE_OPENERS = [
    "remember when",
    "not gonna lie",
    "can we talk about",
    "is it just me or",
    "unpopular opinion",
]

REPETITIVE_OPENERS_FR = [
    "viens de",
    "vient de",
    "jviens de",
    "j'viens de",
    "aujourd'hui",
    "tu te souviens quand",
    "je vais pas mentir",
    "on peut parler de",
    "c'est moi ou",
    "opinion impopulaire",
    "franchement",
]



def tier3d_repetitive_opener(texts, t2_total=0, lang="en"):
    """
    TIER 3d: Repetitive tweet opener detection.
    
    Two-level approach for generalization:
      1. Known bot phrases (hardcoded): count ≥ 3 → 2 pts
         These are proven patterns from observed bot algorithms.
      2. Dynamic opener detection: ANY first-3/4-word prefix repeated ≥ 5 times,
         but only when T2 evidence already exists (T2-gated).
         This catches NEW template bots with openers we haven't seen.
    
    Returns: (score, most_repeated_opener_count)
    """


    if len(texts) < 5:
        return 0, 0

    openers_list = REPETITIVE_OPENERS if lang != "fr" else (REPETITIVE_OPENERS + REPETITIVE_OPENERS_FR)
    
    # Level 1: Known bot opener phrases (proven, zero FP risk)
    max_count = 0
    for opener in openers_list:
        count = sum(1 for t in texts if t.lower().strip().startswith(opener))
        max_count = max(max_count, count)
    
    if max_count >= 3:
        return 2, max_count
    
    # Level 2: Generalized dynamic opener (requires T2 corroboration + higher bar)
    # Catches template bots that reuse any opening phrase, not just the known ones.
    # Higher count threshold (≥5) prevents false positives from topical humans
    # (e.g., sports fans repeating "the game was").
    if t2_total > 0:
        gen_max = 0
        for prefix_len in [3, 4]:
            openers = Counter()
            for t in texts:
                words = t.lower().strip().split()
                if len(words) >= prefix_len:
                    openers[' '.join(words[:prefix_len])] += 1
            if openers:
                _, count = openers.most_common(1)[0]
                gen_max = max(gen_max, count)
        
        if gen_max >= 5:
            return 2, gen_max
    
    return 0, max_count


# --- Tier 3e: Post-Length Uniformity ---
# Bots generated from templates tend to produce tweets with very consistent
# lengths (low coefficient of variation of character count). Humans vary
# naturally between short quips and lengthy threads. This is a weak
# supporting signal (1 pt) that reinforces other evidence.

def tier3e_length_uniform(texts):
    """
    TIER 3e: Post-length uniformity detection.
    
    If the coefficient of variation of post lengths is < 0.30 and there
    are at least 10 posts, award 1 point.
    
    Returns: (score, length_cv)
    """
    if len(texts) < 10:
        return 0, None
    lengths = [len(t) for t in texts]
    mean_len = sum(lengths) / len(lengths)
    if mean_len < 5:
        return 0, None
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std_len = variance ** 0.5
    length_cv = std_len / mean_len
    if length_cv < 0.30:
        return 1, length_cv
    return 0, length_cv


# --- Tier 3f: Human Spam Exemption ---
# Some accounts are human spammers (e.g., porn spam bots) that post
# near-identical content repeatedly. These are NOT AI-generated bots.
# They have extremely high sequential similarity and very low vocabulary.
# Giving them a large negative score ensures they aren't flagged.

def tier3f_spam_exemption(texts):
    """
    TIER 3f: Human spammer exemption.
    
    If posts are near-identical (sequential similarity > 0.75) AND
    vocabulary is extremely low (< 0.20), this is a human spammer,
    not an AI bot. Apply a large negative score.
    
    Returns: (score, is_spam: bool)
    """
    if len(texts) < 5:
        return 0, False
    
    # Sequential similarity
    from difflib import SequenceMatcher
    sim_sum, sim_n = 0, 0
    for i in range(len(texts) - 1):
        if len(texts[i]) > 5 and len(texts[i+1]) > 5:
            sim_sum += SequenceMatcher(None, texts[i], texts[i+1]).ratio()
            sim_n += 1
    avg_sim = sim_sum / sim_n if sim_n > 0 else 0
    
    # Vocabulary ratio
    all_words = ' '.join(texts).lower().split()
    vocab_ratio = len(set(all_words)) / len(all_words) if all_words else 1.0
    
    if avg_sim > 0.75 and vocab_ratio < 0.20:
        return -100, True
    return 0, False


# ============================================================================
# PIPELINE: SCORE A SINGLE USER
# ============================================================================

def score_user(uid, posts, dataset_lang="en", username="", verbose=False):
    """
    Run all detection tiers on a single user and return their total score
    along with a breakdown of which tiers fired.
    """
    posts_sorted = sorted(posts, key=lambda x: x["created_at"])
    texts = [p["text"] for p in posts_sorted]
    times = parse_timestamps(posts_sorted)
    n_posts = len(posts_sorted)

    breakdown = {}
    total_score = 0

    # --- Tier 1: Near-certain signals ---
    s, detail = tier1a_meta_text(texts, lang=dataset_lang)
    breakdown["T1a_meta_text"] = {"score": s, "meta_matches": detail}
    total_score += s

    s, detail = tier1b_encoding_artifacts(texts)
    breakdown["T1b_encoding"] = {"score": s, "garbled_tweets": detail}
    total_score += s

    # --- Tier 2: Strong signals ---
    s, detail = tier2a_same_second_bursts(posts_sorted)
    breakdown["T2a_same_second"] = {"score": s, "duplicate_timestamps": detail}
    total_score += s

    s, detail = tier2b_interval_regularity(times)
    # CV corroboration: the 1.05-1.15 range (3pts) is marginal — require at
    # least 2 @mentions to stay at 3pts.  Users with CV 1.05-1.15 and 0-1
    # mentions are more likely to be regular human posters, not bots.
    # Note: CV 1.00-1.05 is near the 4pt tier boundary and more reliable.
    if s == 3 and detail is not None and detail > 1.05:
        mention_count = sum(1 for t in texts if "@" in t)
        if mention_count < 2:
            s = 2  # fall back to same level as the 1.15-1.20 tier
    breakdown["T2b_interval_cv"] = {"score": s, "cv": detail}
    total_score += s

    s, detail = tier2c_template_pattern(texts, n_posts)
    breakdown["T2c_template"] = {"score": s, "is_template": detail}
    total_score += s

    s, detail = tier2d_just_frequency(texts, n_posts, lang=dataset_lang)
    breakdown["T2d_just_freq"] = {"score": s, "just_rate": detail}
    total_score += s

    s, detail = tier2e_zero_engagement(texts, n_posts)
    breakdown["T2e_zero_engage"] = {"score": s, "is_zero_engagement": detail}
    total_score += s

    # --- Tier 3: Supporting signals ---
    # Some T3 signals are gated on having at least one T2 signal to prevent
    # pure T3 combos from crossing the detection threshold alone.
    t2_total = sum(v["score"] for k, v in breakdown.items() if k.startswith("T2"))

    s, detail = tier3a_hashtag_density(texts)
    breakdown["T3a_hashtags"] = {"score": s, "hashtag_rate": detail}
    total_score += s

    # Low URL rate only counts when a Tier 2 signal has fired.
    # Prevents pure T3 combos like hashtag(2)+low_url(1) from crossing threshold.
    raw_lu_url, lu_url_detail = tier3b_low_url_rate(texts, n_posts)
    lu_url_score = raw_lu_url if t2_total > 0 else 0
    breakdown["T3b_low_url"] = {"score": lu_url_score, "url_rate": lu_url_detail}
    total_score += lu_url_score

    s, detail = tier3c_fun_fact(texts, lang=dataset_lang)
    breakdown["T3c_fun_fact"] = {"score": s, "fun_fact_count": detail}
    total_score += s

    s, detail = tier3d_repetitive_opener(texts, t2_total=t2_total, lang=dataset_lang)
    breakdown["T3d_rep_opener"] = {"score": s, "opener_count": detail}
    total_score += s

    # Length uniformity only counts when a Tier 2 signal has already fired.
    raw_lu, lu_detail = tier3e_length_uniform(texts)
    lu_score = raw_lu if t2_total > 0 else 0
    breakdown["T3e_length_uniform"] = {"score": lu_score, "length_cv": lu_detail}
    total_score += lu_score

    s, detail = tier3f_spam_exemption(texts)
    breakdown["T3f_spam_exempt"] = {"score": s, "is_spam": detail}
    total_score += s

    # --- T1/T2 gating rule ---
    # Require at least one T1 or T2 signal to flag.  T3 signals alone are
    # not reliable enough.  (Currently a no-op safety net — no existing
    # detections rely solely on T3 — but prevents future regressions.)
    t1_total = sum(v["score"] for k, v in breakdown.items() if k.startswith("T1"))
    if t1_total == 0 and t2_total == 0:
        total_score = min(total_score, DETECTION_THRESHOLD - 1)

    # --- Same-second corroboration ---
    # If same-second bursts (+ length_uniform, another weak signal) are the
    # ONLY signals, require ≥6 duplicate timestamps to flag.
    # Exception: very strong length uniformity (len_cv < 0.10) is a genuine
    # corroboration signal — treat it as non-trivial evidence.
    ss_score = breakdown["T2a_same_second"]["score"]
    if ss_score > 0:
        len_score = breakdown["T3e_length_uniform"]["score"]
        len_cv = breakdown["T3e_length_uniform"]["length_cv"]
        # Very low length CV provides genuine corroboration
        if len_cv is not None and len_cv < 0.10:
            corroborating = total_score - ss_score
        else:
            corroborating = total_score - ss_score - len_score
        ss_count = breakdown["T2a_same_second"]["duplicate_timestamps"]
        if corroborating <= 0 and ss_count < 6:
            total_score = min(total_score, 2)

    if verbose:
        flagged = total_score >= DETECTION_THRESHOLD
        status = "🤖 BOT" if flagged else "   human"
        print(f"  {status} | score={total_score:3d} | {uid[:8]}... | {n_posts:3d} tweets", end="")
        fired = [k for k, v in breakdown.items() if v["score"] > 0]
        if fired:
            print(f" | fired: {', '.join(fired)}", end="")
        print()

    return total_score, breakdown


# ============================================================================
# PIPELINE: PROCESS FULL DATASET
# ============================================================================

def detect_bots(dataset_path, threshold=DETECTION_THRESHOLD, verbose=False):
    """
    Load a dataset, score all users, and return the list of detected bot IDs.
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # Extract metadata
    ds_id = data.get("id", "?")
    lang = data.get("lang", "?")
    metadata = data.get("metadata", {})
    total_users = metadata.get("total_amount_users", len(data.get("users", [])))
    total_posts = metadata.get("total_amount_posts", len(data.get("posts", [])))

    print(f"Dataset {ds_id} | Language: {lang}")
    print(f"Users: {total_users} | Posts: {total_posts}")
    print(f"Detection threshold: {threshold}")
    print("-" * 60)

    # Group posts by author
    user_posts = defaultdict(list)
    for p in data["posts"]:
        user_posts[p["author_id"]].append(p)

    # Map user ID to username
    user_objects = {u["id"]: u for u in data.get("users", [])}

    # Score each user
    detections = []
    all_scores = {}

    for uid, posts in user_posts.items():
        username = user_objects.get(uid, {}).get("username", "")
        score, breakdown = score_user(uid, posts, dataset_lang=lang, username=username, verbose=verbose)
        all_scores[uid] = (score, breakdown)
        if score >= threshold:
            detections.append(uid)

    # Summary
    print("-" * 60)
    print(f"Detected {len(detections)} bots out of {len(user_posts)} users")
    print(f"Detection rate: {len(detections)/len(user_posts):.1%}")

    # Score distribution
    score_vals = [s for s, _ in all_scores.values()]
    bins = Counter()
    for s in score_vals:
        if s == 0:
            bins["0"] += 1
        elif s < threshold:
            bins[f"1-{threshold-1}"] += 1
        else:
            bins[f">={threshold}"] += 1
    print(f"Score distribution: {dict(bins)}")

    return detections, all_scores


# ============================================================================
# EVALUATION (for practice datasets with known bots)
# ============================================================================

def evaluate(detections, bots_file):
    """
    Evaluate detections against known bot labels.
    Uses the competition scoring: +4 TP, -1 FN, -2 FP.
    """
    with open(bots_file, "r", encoding="utf-8-sig") as f:
        true_bots = set(line.strip() for line in f if line.strip())

    detected = set(detections)
    tp = len(detected & true_bots)
    fp = len(detected - true_bots)
    fn = len(true_bots - detected)
    tn = -1  # We don't track total users here

    score = 4 * tp - 1 * fn - 2 * fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / len(true_bots) if true_bots else 0
    max_score = 4 * len(true_bots)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"True bots in dataset:  {len(true_bots)}")
    print(f"Detected as bots:      {len(detected)}")
    print(f"True Positives (TP):   {tp}")
    print(f"False Positives (FP):  {fp}")
    print(f"False Negatives (FN):  {fn}")
    print(f"Precision:             {precision:.1%}")
    print(f"Recall:                {recall:.1%}")
    print(f"Competition Score:     {score:+d} / {max_score} ({score/max_score:.0%} of perfect)")
    print("=" * 60)

    # Show missed bots
    missed = true_bots - detected
    if missed:
        print(f"\nMissed bots ({len(missed)}):")
        for uid in sorted(missed):
            print(f"  {uid}")

    # Show false positives
    false_pos = detected - true_bots
    if false_pos:
        print(f"\nFalse positives ({len(false_pos)}):")
        for uid in sorted(false_pos):
            print(f"  {uid}")

    return {"tp": tp, "fp": fp, "fn": fn, "score": score, "max": max_score}


# ============================================================================
# OUTPUT: Write detection file in competition format
# ============================================================================

def write_detections(detections, output_path):
    """Write detected bot IDs to a file, one per line."""
    with open(output_path, "w") as f:
        for uid in sorted(detections):
            f.write(uid + "\n")
    print(f"\nDetections written to {output_path} ({len(detections)} accounts)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bot or Not Challenge — Bot Detection Pipeline"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help=(
            "Path to dataset_posts_users.json file (or alias: 30/32/all). "
            "Default: all (runs both dataset 30 and 32)"
        ),
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=DETECTION_THRESHOLD,
        help=f"Detection score threshold (default: {DETECTION_THRESHOLD})",
    )
    parser.add_argument(
        "--bots", "-b",
        help=(
            "Path to dataset.bots.txt for evaluation (optional). "
            "If omitted, auto-uses matching dataset.bots.<id>.txt when possible"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path for detections (default: detections.txt)",
        default="detections.txt",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-user scoring details",
    )

    args = parser.parse_args()

    dataset_arg = str(args.dataset).strip().lower()
    if dataset_arg in {"all", "both"}:
        dataset_paths = [
            str(DATASET_POSTS_USERS_30),
            str(DATASET_POSTS_USERS_31),
            str(DATASET_POSTS_USERS_32),
            str(DATASET_POSTS_USERS_33),
        ]
    else:
        dataset_paths = [_resolve_dataset_alias(args.dataset)]

    run_summaries = []

    for dataset_path in dataset_paths:
        print("\n" + "#" * 60)
        print(f"RUNNING DATASET: {Path(dataset_path).name}")
        print("#" * 60)

        # Run detection
        detections, all_scores = detect_bots(
            dataset_path,
            threshold=args.threshold,
            verbose=args.verbose,
        )

        # Write output (dataset-specific names when running multiple datasets)
        output_path = args.output
        if len(dataset_paths) > 1:
            output_path = _output_path_for_dataset(args.output, dataset_path)
        write_detections(detections, output_path)

        # Evaluate if ground truth is provided/available
        bots_path = args.bots or _default_bots_for_dataset(dataset_path)
        eval_result = None
        if bots_path:
            eval_result = evaluate(detections, bots_path)

        run_summaries.append(
            {
                "dataset": Path(dataset_path).name,
                "users": len(all_scores),
                "detections": len(detections),
                "eval": eval_result,
            }
        )

    if len(run_summaries) > 1:
        print("\n" + "=" * 60)
        print("FINAL SUMMARY (ALL DATASETS)")
        print("=" * 60)

        total_detections = sum(r["detections"] for r in run_summaries)
        total_users = sum(r["users"] for r in run_summaries)
        print(f"Total detections:      {total_detections}")
        print(f"Total users scanned:   {total_users}")
        print(f"Overall detection rate:{(total_detections / total_users):.1%}")

        eval_runs = [r for r in run_summaries if r["eval"] is not None]
        if eval_runs:
            total_tp = sum(r["eval"]["tp"] for r in eval_runs)
            total_fp = sum(r["eval"]["fp"] for r in eval_runs)
            total_fn = sum(r["eval"]["fn"] for r in eval_runs)
            total_score = sum(r["eval"]["score"] for r in eval_runs)
            total_max = sum(r["eval"]["max"] for r in eval_runs)
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

            print(f"Combined TP/FP/FN:     {total_tp} / {total_fp} / {total_fn}")
            print(f"Combined Precision:    {precision:.1%}")
            print(f"Combined Recall:       {recall:.1%}")
            print(
                f"Combined Score:        {total_score:+d} / {total_max} "
                f"({(total_score / total_max):.0%} of perfect)"
            )

        print("=" * 60)


if __name__ == "__main__":
    main()