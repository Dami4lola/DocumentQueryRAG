/** File-type helpers for uploads. Pure functions, safe to use in the browser or Node. */

export type DocumentKind = "pdf" | "docx";

export const DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
export const ACCEPTED_FILES = `.pdf,.docx,application/pdf,${DOCX_MIME}`;

/** Roughly one printed page; Word files have no reliable page breaks in their text. */
export const PSEUDO_PAGE_CHARS = 3000;

export function documentKind(name: string, mimeType = ""): DocumentKind | null {
  const lower = name.toLowerCase();
  if (lower.endsWith(".pdf") || mimeType === "application/pdf") return "pdf";
  if (lower.endsWith(".docx") || mimeType === DOCX_MIME) return "docx";
  return null;
}

/** Groups paragraphs into pages of about PSEUDO_PAGE_CHARS characters. */
export function splitIntoPages(text: string): string[] {
  const pages: string[] = [];
  let current = "";
  const flush = () => {
    if (current.trim()) pages.push(current.trim());
    current = "";
  };

  for (const paragraph of text.split(/\n\s*\n/)) {
    if (!paragraph.trim()) continue;
    if (current && current.length + paragraph.length + 2 > PSEUDO_PAGE_CHARS) flush();
    current = current ? `${current}\n\n${paragraph}` : paragraph;
    // A single paragraph longer than a page is cut into page-sized pieces.
    while (current.length > PSEUDO_PAGE_CHARS) {
      pages.push(current.slice(0, PSEUDO_PAGE_CHARS));
      current = current.slice(PSEUDO_PAGE_CHARS);
    }
  }
  flush();
  return pages;
}
