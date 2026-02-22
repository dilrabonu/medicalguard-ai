def format_bilingual_header(lang: str) -> str:
    if lang == "uz":
        return " **PharmScan AI — Dori yorlig‘i skriningi**"
    return " **PharmScan AI — Medication label screening**"


def format_answer(drug: str, lang: str, context_chunks: list[dict], decision_block: str | None = None) -> str:
    """
    Produce a safe, user-friendly answer using RAG context.
    No dosing. Always include a safety note.
    """
    header = format_bilingual_header(lang)
    ctx = "\n\n".join([f"- {c['text']}" for c in context_chunks]) if context_chunks else "- (No KB context found yet)"

    if lang == "uz":
        return f"""
{header}

**Dori nomi:** {drug}

{decision_block or ""}

### Qisqacha ma’lumot (KB asosida)
{ctx}

### Qanday foydalaniladi? (umumiy)
- Faqat yo‘riqnoma va shifokor/farmatsevt ko‘rsatmalariga amal qiling.
- Agar natija **REVIEW/REJECT** bo‘lsa: rasmni qayta oling, yorug‘likni yaxshilang, farmatsevtga ko‘rsating.

### Xavfsizlik eslatmasi
Bu tizim **skrining** uchun. U dori haqiqiyligini kafolatlamaydi va tibbiy maslahat bermaydi.
""".strip()
    else:
        return f"""
{header}

**Drug name:** {drug}

{decision_block or ""}

### Summary (from KB)
{ctx}

### How to use (general)
- Follow the leaflet and your pharmacist/clinician instructions.
- If the decision is **REVIEW/REJECT**: re-take the photo with better lighting and ask a pharmacist.

### Safety note
This is a **screening** tool. It does not verify authenticity and is not medical advice.
""".strip()
