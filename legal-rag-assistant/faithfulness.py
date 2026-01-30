def legal_faithfulness(answer: str, chunks: list) -> float:
    if not chunks or not answer:
        return 0.0

    hits = 0
    for c in chunks:
        sec = c.get("section_number")
        if sec and sec in answer:
            hits += 1

    return round(hits / max(len(chunks), 1), 3)