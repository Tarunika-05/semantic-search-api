import re
import hashlib
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

# ─────────────────────────────────────────────────────────────────────
# Multi-Stage Preprocessing Pipeline
#
# The 20 Newsgroups corpus was collected from Usenet — a 1990s-era
# bulletin board system. Even after sklearn strips headers/footers/
# quotes, the surviving text is full of structural noise:
#
#   - URLs (http://, ftp://) that bias embeddings toward web junk
#   - Email addresses that encode identity, not topic
#   - File paths (/usr/lib, C:\WINDOWS) that are OS artifacts
#   - Excessive whitespace from quoted-reply indentation remnants
#   - Cross-posted duplicates that inflate cluster sizes
#
# Each preprocessing stage below targets a specific noise source.
# The comments explain *why* each step matters for downstream
# embedding quality and clustering accuracy.
# ─────────────────────────────────────────────────────────────────────

# Compiled regex patterns — defined once, reused across all documents.
# Pre-compilation avoids re-parsing the regex on every document.

# URLs — matches http://, https://, ftp:// and www. prefixed links.
# These are structural artifacts of email communication, not topical content.
# An embedding model would learn spurious associations between URLs and topics.
_URL_PATTERN = re.compile(r'https?://\S+|ftp://\S+|www\.\S+', re.IGNORECASE)

# Email addresses — encode sender/recipient identity, not discussion topic.
# A post about gun control should not embed closer to other posts by the
# same author just because they share an email address.
_EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')

# File paths — OS-level artifacts (e.g. /usr/lib/X11, C:\WINDOWS\system32).
# Common in comp.* newsgroups. They carry no semantic meaning for topic
# classification and would create false similarity between unrelated posts
# that happen to mention the same directory structure.
_FILEPATH_PATTERN = re.compile(r'(?:[A-Za-z]:\\[\w\\.-]+|/(?:usr|etc|var|home|tmp|bin|lib|opt|dev)[\w/.-]*)')

# Whitespace normalisation — collapse runs of spaces/tabs/newlines.
# After stripping quotes, many posts have 5-10 blank lines in a row
# from indentation artifacts. These waste embedding capacity on nothing.
_WHITESPACE_PATTERN = re.compile(r'\s+')

# Non-printable / control characters — binary junk from encoding issues.
# Some 1990s posts contain garbled characters from charset mismatches.
_CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def _clean_text(text: str) -> str:
    """
    Apply regex-based cleaning to a single document.

    Order matters:
    1. Remove URLs first (they contain @ and / which would partially
       match email and filepath patterns if processed later)
    2. Remove emails
    3. Remove file paths
    4. Strip control characters
    5. Normalise whitespace last (after removing items that create gaps)

    Returns:
        Cleaned text string, stripped of structural noise.
    """
    text = _URL_PATTERN.sub(' ', text)
    text = _EMAIL_PATTERN.sub(' ', text)
    text = _FILEPATH_PATTERN.sub(' ', text)
    text = _CONTROL_CHARS_PATTERN.sub('', text)
    text = _WHITESPACE_PATTERN.sub(' ', text)
    return text.strip()


def _compute_fingerprint(text: str) -> str:
    """
    Compute a content fingerprint for near-duplicate detection.

    Strategy: extract character trigrams, sort them, hash the result.
    This is a lightweight alternative to full MinHash/LSH. It catches:
    - Exact duplicates (same post cross-posted to multiple newsgroups)
    - Near-duplicates (same post with minor formatting differences)

    Why trigrams, not full text hash?
    A full MD5 of the raw text would miss near-duplicates — a single
    extra space would produce a completely different hash. Trigram-based
    fingerprinting is robust to minor whitespace/punctuation differences.

    I normalise to lowercase and strip non-alphanumeric characters first,
    so "Hello World!" and "hello world" produce the same fingerprint.
    """
    # Normalise: lowercase, keep only alphanumeric + spaces
    normalised = re.sub(r'[^a-z0-9 ]', '', text.lower())
    # Extract sorted character trigrams
    trigrams = sorted(set(
        normalised[i:i+3] for i in range(len(normalised) - 2)
    ))
    # Hash the trigram set — deterministic fingerprint
    fingerprint = hashlib.md5(''.join(trigrams).encode()).hexdigest()
    return fingerprint


def load_documents():
    """
    Load and preprocess the 20 Newsgroups dataset with a multi-stage pipeline.

    Preprocessing decisions:
    ─────────────────────────────────────────────────────────────────────
    1. remove=("headers", "footers", "quotes")
       - Headers contain email metadata (From:, Subject:, NNTP-Posting-Host:)
         which are not semantically meaningful and would bias embeddings
         toward author/server identity rather than topic content.
       - Footers contain email signatures, disclaimers, and contact info —
         again, noise that does not reflect document semantics.
       - Quotes are reply chains from previous emails. Including them would
         cause documents to absorb topics from other posts they're replying
         to, making cluster boundaries artificially fuzzy in the wrong way.

    2. subset="train"
       - I use only the training split to keep the corpus consistent
         and avoid any accidental overlap with evaluation data.

    3. documents[:5000]
       - The full dataset has ~11,000 training documents. I cap at 5000
         to allow rapid experimentation while preserving topic diversity
         across all 20 categories (~250 docs per category on average).
       - This is a deliberate trade-off: speed and iteration vs. coverage.
         For production, the full corpus would be used.

    Pipeline stages:
       Stage 1 → Regex cleaning (URLs, emails, file paths, whitespace)
       Stage 2 → Length filtering (remove sub-50-char documents)
       Stage 3 → Near-duplicate removal (trigram fingerprinting)
    ─────────────────────────────────────────────────────────────────────

    Returns:
        documents:   list of cleaned text strings
        labels:      list of integer category labels
        label_names: list of category name strings
    """

    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes")
    )

    raw_documents = data.data[:5000]
    raw_labels = data.target[:5000]
    label_names = data.target_names

    total_raw = len(raw_documents)

    # ── Stage 1: Regex cleaning ──
    # Apply multi-pattern cleaning to strip structural noise.
    # This runs before length filtering because URL/email removal
    # can reduce a document's meaningful content below the threshold.
    cleaned_pairs = [
        (_clean_text(doc), label)
        for doc, label in zip(raw_documents, raw_labels)
    ]

    # ── Stage 2: Length filtering ──
    # After stripping noise, some posts become empty or near-empty.
    # Sub-50-char documents produce meaningless embeddings that sit
    # near the origin in embedding space, polluting clusters.
    MIN_DOC_LENGTH = 50
    length_filtered = [
        (doc, label) for doc, label in cleaned_pairs
        if len(doc.strip()) > MIN_DOC_LENGTH
    ]
    removed_by_length = len(cleaned_pairs) - len(length_filtered)

    # ── Stage 3: Near-duplicate removal ──
    # Newsgroups have cross-posts (same message posted to multiple groups)
    # and reply chains that survive quote stripping. Duplicates artificially
    # inflate certain cluster sizes and bias the GMM toward over-representing
    # popular discussion threads rather than topics.
    seen_fingerprints = set()
    deduplicated = []
    for doc, label in length_filtered:
        fp = _compute_fingerprint(doc)
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            deduplicated.append((doc, label))
    removed_by_dedup = len(length_filtered) - len(deduplicated)

    documents, labels = zip(*deduplicated) if deduplicated else ([], [])

    # ── Preprocessing stats ──
    # Transparency: show exactly what the pipeline did so the reader
    # can evaluate whether the filtering is too aggressive or too lenient.
    print(f"\n{'─'*55}")
    print("📊 PREPROCESSING PIPELINE STATS")
    print(f"{'─'*55}")
    print(f"   Raw documents loaded:      {total_raw}")
    print(f"   After regex cleaning:       {len(cleaned_pairs)} (cleaned in-place)")
    print(f"   Removed by length (<{MIN_DOC_LENGTH} chars): {removed_by_length}")
    print(f"   Removed as duplicates:      {removed_by_dedup}")
    print(f"   Final corpus size:          {len(documents)}")
    print(f"   Categories:                 {len(label_names)}")
    print(f"{'─'*55}")

    # Category distribution — check that I haven't accidentally
    # wiped out entire categories during filtering.
    cat_counts = Counter(labels)
    print(f"   Category distribution:")
    for cat_id, count in sorted(cat_counts.items()):
        print(f"     {label_names[cat_id][:30]:<30s} {count:>4d} docs")
    print(f"{'─'*55}")

    print(f"\n✅ Loaded {len(documents)} documents after preprocessing.")
    print(f"   Example doc preview: {documents[0][:100]!r}")

    return list(documents), list(labels), label_names
