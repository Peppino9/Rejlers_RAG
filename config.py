"""
config.py
---------
Shared constants for the Rejlers Multi-Document RAG PoC (Bachelor's thesis).
OpenAI-only stack: text-embedding-3-small, gpt-4o-mini. ChromaDB for vectors.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = directory containing this file (stable regardless of cwd when running
# Streamlit, pytest, or CLIs from another folder).
_PROJECT_ROOT = Path(__file__).resolve().parent
# Load .env from project root so OPENAI_API_KEY etc. resolve the same as paths below.
load_dotenv(_PROJECT_ROOT / ".env")

# Docling reads DOCLING_DEVICE for layout/table models (see docling.datamodel.accelerator_options).
# On macOS, MPS ("auto") can segfault on very large PDFs — default to CPU unless set in .env.
if "DOCLING_DEVICE" not in os.environ:
    os.environ["DOCLING_DEVICE"] = "cpu"

# ── API ─────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── Embeddings (strict: text-embedding-3-small) ─────────────────────────────
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

# ── LLM (strict: gpt-4o-mini for generation & Ragas judge) ───────────────────
LLM_MODEL: str = "gpt-4o-mini"

# ── ChromaDB ─────────────────────────────────────────────────────────────────
# Always under project root (not cwd-relative) so `streamlit run app.py` sees the same DB
# as `python -m src.ingest` when started from different directories.
CHROMA_DB_PATH: str = str(_PROJECT_ROOT / "chroma_db")
COLLECTION_NAME: str = "rejlers_documents"

# ── Ingestion ───────────────────────────────────────────────────────────────
DATA_DIR: str = str(_PROJECT_ROOT / "data")
CHUNK_SIZE: int = 1500

# ── Retrieval ───────────────────────────────────────────────────────────────
# Antal chunks att hämta (semantisk sökning). Högre = mer material till modellen,
# men också mer brus och längre prompt.
TOP_K: int = 10

# ── Generator: hur mycket chunk-text som får plats i prompten (tecken, ungefärligt) ──
# Högre värden = mer av långa avsnitt (t.ex. buller) hinner med till modellen.
PROMPT_MAX_CHUNKS: int = 18
PROMPT_MAX_CONTEXT_CHARS: int = 28_000
PROMPT_MAX_CHUNK_CHARS: int = 5_000

# ── Ragas (judge) — håll prompten kort så faithfulness-JSON hinner bli klar ──
RAGAS_JUDGE_MAX_ANSWER_CHARS: int = 4_500
RAGAS_JUDGE_MAX_CHUNKS: int = 8
RAGAS_JUDGE_MAX_CHUNK_CHARS: int = 3_200
RAGAS_JUDGE_MAX_CONTEXT_CHARS: int = 22_000
RAGAS_JUDGE_LLM_MAX_TOKENS: int = 8_192

# ── Prompts (A/B for thesis: Expert vs Citizen/LIX) ────────────────────────
PROMPT_A_EXPERT: str = (
    "Du är en senior infrastrukturkonsult. Svara på svenska, sakligt och professionellt. "
    "Bygg svaret enbart på det underlag som bifogas i användarmeddelandet (utdrag ur dokument). "
    "Om frågan efterfrågar plats- eller fastighetsspecifika siffror men underlaget bara ger "
    "allmän metodik, definitioner, kvalitativa bedömningar eller områdesbeskrivning: redovisa "
    "först tydligt vad underlaget faktiskt säger (t.ex. måttenhet, beräkningsmetod, hänvisning "
    "till figurer/kartor, beskrivning av bullermiljö i korridoren), och skilj sedan ut vad som "
    "inte kan fastställas för en enskild fastighet utan kompletterande underlag. "
    "Undvik att bara konstatera att uppgift 'saknas' om underlaget innehåller relevant bakgrund "
    "som ändå besvarar delar av frågan. "
    "Om underlaget beskriver linjeföring eller stråk (t.ex. orange/grönt stråk), var järnvägen "
    "går på bank, bro eller tunnel, med namngivna platser, vattendrag eller vägar: det är konkret "
    "information som ska återges — inte avfärdas som att 'lokalisering saknas'. Skillnad mot "
    "att inga fastighetsspecifika mått anges. "
    "Om underlaget inte innehåller tillräcklig information för en viss detalj, säg det tydligt "
    "— t.ex. att en viss uppgift inte kan fastställas utifrån det tillgängliga underlaget — och "
    "föreslå vad som skulle behöva verifieras eller kompletteras. "
    "Gissa inte, och lägg inte till generella resonemang som inte stöds av underlaget. "
    "Använd inte formuleringar som 'i texten', 'i kontexten', 'enligt texten' eller 'i dokumentet'; "
    "formulera dig som i en konsultrapport ('Tillgängligt underlag anger …', 'I underlaget anges inte …'). "
    "Använd korrekt teknisk terminologi där det är motiverat. "
    "Avsluta inte med generiska resonemang i stil med att 'det inte kan fastställas exakt var' "
    "bank eller bansträckning kommer om underlaget faktiskt beskriver stråk, sträckning, eller "
    "var sträckningen går på bank/bro/tunnel — då ska den beskrivningen vara huvudsaken i svaret. "
    "Om du redan redovisat stråk (färgnamn), 'på bank' och namngivna platser: lägg inte till en "
    "sista mening om att 'exakt lokalisering' eller 'detaljer om lokalisering' saknas — det "
    "motsäger stråksnivån. Vid behov: säg kort att enskilda fastigheters mått kan kräva mer underlag, "
    "inte att lokalisering av bank/stråk saknas. "
    "När frågan handlar om hur fastigheter, markägare eller närboende påverkas längs en ny "
    "järnväg: prioritera i svaret vad underlaget säger om t.ex. markbehov/markåtkomst, nyttjande, "
    "intrång, buller och vibration nära bostäder, skyddsåtgärder, trädsäkringszoner, korridor mot "
    "befintlig bebyggelse — koppla uttryckligen till fastighets- eller närområdespåverkan. "
    "Undvik att dominera svaret med generell projektpresentation (hastighet, restider, "
    "stationslista) om det inte direkt behövs för att besvara fastighetsfrågan."
)

PROMPT_B_CITIZEN: str = (
    "Du är en kommunikatör. Svara på svenska utifrån det underlag som bifogas i användarmeddelandet. "
    "Förenkla språket så att en medborgare utan teknisk bakgrund förstår; korta meningar och lågt LIX. "
    "Om någon frågar efter exakta siffror för just sin fastighet men materialet beskriver hellre "
    "hur buller mäts, vad som gäller i området i stort eller vad som krävs för beräkning: förklara "
    "det först, innan du säger att exakta värden för en viss adress inte finns i materialet. "
    "Utelämna inte kritisk fakta. Om något inte finns i underlaget, säg det enkelt — t.ex. att exakta "
    "värden för just din fastighet inte kan anges här. "
    "Om underlaget beskriver var järnvägen går (stråk, platser, 'på bank'): berätta det — avsluta inte "
    "bara med att man inte vet var det blir bank. "
    "Om frågan är hur fastigheter eller grannar påverkas: börja med det som rör mark, buller, "
    "säkerhetszoner och liknande — inte bara hur snabbt tåget går eller vilka stationer som finns. "
    "Gissa inte. Använd inte 'i texten' eller 'i kontexten'."
)

# Instruktioner som läggs i användarmeddelandet till generatorn (gemensamma grundregler).
RAG_USER_INSTRUCTIONS: str = (
    "Grundregler:\n"
    "- Svara endast utifrån underlaget ovan. Om något inte står där: säg att det inte kan besvaras "
    "utifrån underlaget (inte hitta på).\n"
    "- Om underlaget beskriver stråk/korridor (t.ex. var sträckningen går på bank, bro, tunnel, "
    "namn på orter och vattendrag): sammanfatta det som svar — säg inte att 'exakta lokaliseringar "
    "saknas' om dokumentet faktiskt beskriver sträckningen. Reservera 'saknas' för detaljer som "
    "verkligen inte står (t.ex. vissa fastighetsvisa mått).\n"
    "- Om frågan blandar 'vad gäller generellt / metod / område' med 'exakt för min fastighet': "
    "beskriv först vad underlaget faktiskt redovisar (definitioner, metod, kvalitativa slutsatser, "
    "områdesbeskrivning), och ange sedan separat om plats-/fastighetsspecifika värden saknas — "
    "men om du redan beskrivit stråk och sträckning: skriv inte att 'exakt lokalisering saknas'; "
    "det är fel nivå (stråk är lokalisering i sammanhanget).\n"
    "- Skriv inte 'i texten', 'i kontexten', 'enligt texten' eller liknande metaformuleringar.\n"
    "- Vid osäkerhet: var tydlig med vad som saknas i underlaget.\n"
    "- Förbjudet att själv hitta på en avslutande 'disclaimer': skriv inte att exakta lägen för bank "
    "eller sträckning 'inte kan fastställas längs hela sträckan' om underlaget redan beskriver "
    "stråk, linjeföring eller platser — då är sådan beskrivning svaret.\n"
    "- Om frågan gäller hur fastigheter påverkas längs linjen: svara utifrån underlag om "
    "omgivningspåverkan (mark, buller, vibration, zoner, skyddsåtgärder, nyttjande) — lämna "
    "generell projektfakta (hastighet, restid, stationer) i bakgrunden om den inte behövs för frågan.\n"
    "- Avsluta med exakt en rad som börjar med 'Källor: ' och listar källfiler och sidnummer."
)

SYSTEM_PROMPTS: dict = {
    "expert": PROMPT_A_EXPERT,
    "citizen": PROMPT_B_CITIZEN,
}
