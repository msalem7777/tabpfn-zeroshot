"""Credential-free scholarly search and bounded paper retrieval."""
import io
import json
from urllib.parse import quote, urlencode, urljoin

from bs4 import BeautifulSoup

from .evidence import source_record
from .network import get_bytes


def text_from_bytes(data, content_type="", filename=""):
    if "pdf" in content_type.lower() or filename.lower().endswith(".pdf") or data.startswith(b"%PDF"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:100])
        if not text.strip():
            raise ValueError("This PDF has no extractable text; supply a text version (OCR is not included).")
    else:
        soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
        for node in soup(["script", "style", "nav", "footer"]):
            node.decompose()
        text = soup.get_text(" ", strip=True)
    return text[:180000]


def retrieve(url):
    data, kind, final_url = get_bytes(url)
    text = text_from_bytes(data, kind, final_url)
    notes = []
    # Publisher landing pages often omit tables. Follow explicit article PDF
    # metadata or PDF links, with the same public-URL checks as every download.
    if "pdf" not in kind.lower() and not data.startswith(b"%PDF"):
        soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
        links = [m.get("content", "") for m in soup.select('meta[name="citation_pdf_url"]')]
        links += [a.get("href", "") for a in soup.select("a[href]")
                  if a.get("href", "").lower().split("?")[0].endswith(".pdf")
                  or a.get_text(" ", strip=True).lower() in ("pdf", "download pdf", "view pdf")]
        for link in list(dict.fromkeys(links))[:2]:
            try:
                pdf, pdf_kind, pdf_url = get_bytes(urljoin(final_url, link))
                if not pdf.startswith(b"%PDF"):
                    continue
                pdf_text = text_from_bytes(pdf, pdf_kind, pdf_url)
                if len(pdf_text) > len(text):
                    text, final_url = pdf_text, pdf_url
                    break
            except ValueError as exc:
                notes.append("Linked PDF unavailable: " + str(exc))
    if len(text.strip()) < 200:
        raise ValueError("Insufficient retrieved text (under 200 characters); upload a readable PDF or paste the methods and tables.")
    return source_record(final_url, text, final_url, "retrieved", retrieval_notes=notes)


def search(query, limit=6):
    """Search two real indexes; retain partial success and explicit errors."""
    results, errors = [], []
    try:
        url = "https://api.crossref.org/works?" + urlencode({"query.bibliographic": query, "rows": limit})
        data = json.loads(get_bytes(url)[0])
        for x in data["message"]["items"]:
            doi = x.get("DOI", "")
            results.append({"title": (x.get("title") or [doi])[0], "url": "https://doi.org/" + doi,
                            "doi": doi, "abstract": text_from_bytes(x.get("abstract", "").encode()),
                            "index": "Crossref", "fulltext_url": ""})
    except (ValueError, KeyError) as exc:
        errors.append(f"Crossref: {exc}")
    try:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode(
            {"query": query, "format": "json", "resultType": "core", "pageSize": limit})
        data = json.loads(get_bytes(url)[0])
        for x in data.get("resultList", {}).get("result", []):
            pmcid = x.get("pmcid")
            results.append({"title": x.get("title", "Untitled"), "doi": x.get("doi", ""),
                            "url": "https://europepmc.org/article/" + x.get("source", "MED") + "/" + x["id"],
                            "abstract": x.get("abstractText", ""), "index": "Europe PMC",
                            "fulltext_url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/{quote(pmcid)}/fullTextXML" if pmcid and x.get("isOpenAccess") == "Y" else ""})
    except (ValueError, KeyError) as exc:
        errors.append(f"Europe PMC: {exc}")
    unique = {}
    for item in results:
        key = item["doi"].lower() or item["title"].lower()
        # Prefer an open full text link when both indexes find the same study.
        if key not in unique or item["fulltext_url"]:
            unique[key] = item
    return {"results": list(unique.values()), "errors": errors, "query": query}


def collect(items, limit=8):
    sources, errors = [], []
    for item in items[:limit]:
        try:
            s = retrieve(item.get("fulltext_url") or item["url"])
            s.update(title=item["title"], doi=item.get("doi", ""))
            sources.append(s)
        except ValueError as exc:
            errors.append(f"{item['title']}: {exc}")
            if item.get("abstract"):
                sources.append(source_record(item["title"], item["abstract"], item["url"], "abstract",
                                             doi=item.get("doi", "")))
    return sources, errors
