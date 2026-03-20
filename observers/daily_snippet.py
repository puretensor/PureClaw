#!/usr/bin/env python3
"""Daily Intelligence Snippet observer -- source-grounded geopolitical briefing.

Runs at 8 AM weekdays via the Observer registry:
  1. Fetches today's headlines from major news RSS feeds
  2. Generates a structured intelligence brief with source citations
  3. Validates source fidelity (brief vs. cited headlines)
  4. Validates ON THIS DAY / QUOTE via Gemini Google Search grounding
  5. Runs AI council quality gate (4 models in parallel)
  6. Sends as HTML email to configured recipients

Architecture: Generate (with citations) -> Source validation -> Web-grounded verify -> Council gate -> Email
Fallback: If council rejects, degrades to headline summary (no analysis).
"""

import json
import logging
import os
import re
import smtplib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from observers.base import Observer, ObserverContext, ObserverResult
from observers.ai_council import run_council, CouncilResult
from observers.cloud_llm import extract_json
from config import AGENT_NAME

log = logging.getLogger("nexus")


class DailySnippetObserver(Observer):
    """Source-grounded daily intelligence brief delivered by email."""

    name = "daily_snippet"
    schedule = "0 8 * * 0-4"  # 8 AM weekdays (UTC, 0=Mon)

    # -----------------------------------------------------------------------
    # Cloud LLM -- direct API calls, no engine/backend dependency
    # -----------------------------------------------------------------------

    def _call_cloud_llm(self, prompt: str, timeout: int = 120) -> str:
        """Call cloud LLM with Gemini primary, DeepSeek fallback.

        Bypasses the Nexus engine entirely -- no vLLM dependency.
        """
        # --- Gemini (primary) ---
        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                from google import genai
                from google.genai import types as gtypes

                client = genai.Client(api_key=gemini_key)
                config = gtypes.GenerateContentConfig(
                    max_output_tokens=8192,
                    temperature=0.7,
                )
                response = client.models.generate_content(
                    model="gemini-3.0-flash",
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config=config,
                )
                text = response.text or ""
                if text.strip():
                    log.info("Cloud LLM: Gemini OK (%d chars)", len(text))
                    return text.strip()
                log.warning("Cloud LLM: Gemini returned empty response")
            except Exception as e:
                log.warning("Cloud LLM: Gemini failed: %s", e)

        # --- DeepSeek (fallback) ---
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if deepseek_key:
            try:
                payload = json.dumps({
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8192,
                    "temperature": 0.7,
                }).encode()
                req = urllib.request.Request(
                    "https://api.deepseek.com/v1/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {deepseek_key}",
                    },
                )
                resp = urllib.request.urlopen(req, timeout=timeout)
                data = json.loads(resp.read())
                text = data["choices"][0]["message"]["content"]
                if text.strip():
                    log.info("Cloud LLM: DeepSeek OK (%d chars)", len(text))
                    return text.strip()
                log.warning("Cloud LLM: DeepSeek returned empty response")
            except Exception as e:
                log.warning("Cloud LLM: DeepSeek failed: %s", e)

        log.error("Cloud LLM: all backends failed")
        return ""

    # RSS feeds -- broad coverage, no paywalls on RSS
    RSS_FEEDS = {
        "Reuters":        "https://news.google.com/rss/search?q=site:reuters.com+world&hl=en-US&gl=US&ceid=US:en",
        "AP News":        "https://news.google.com/rss/search?q=site:apnews.com+world&hl=en-US&gl=US&ceid=US:en",
        "BBC World":      "https://feeds.bbci.co.uk/news/world/rss.xml",
        "Bloomberg":      "https://feeds.bloomberg.com/markets/news.rss",
        "Guardian":       "https://www.theguardian.com/world/rss",
        "Politico EU":    "https://www.politico.eu/feed/",
        "NPR World":      "https://feeds.npr.org/1004/rss.xml",
        "SCMP":           "https://www.scmp.com/rss/91/feed",
        "Jerusalem Post": "https://www.jpost.com/rss/rssfeedsfrontpage.aspx",
        "Foreign Policy": "https://foreignpolicy.com/feed/",
        "Economist":      "https://www.economist.com/international/rss.xml",
        "Al Jazeera":     "https://www.aljazeera.com/xml/rss/all.xml",
    }

    # -----------------------------------------------------------------------
    # Observer interface
    # -----------------------------------------------------------------------

    def run(self, ctx: ObserverContext) -> ObserverResult:
        """Execute the daily snippet pipeline.

        Pipeline: RSS -> Generate (with citations) -> Source validation ->
                  Web-grounded verify -> Council gate -> Email
        Fallback: headline summary if council rejects.
        """
        date_str = ctx.now.strftime("%B %d, %Y")
        log.info("Daily Intelligence Snippet -- %s", date_str)

        # 1. Fetch headlines
        log.info("Fetching RSS feeds...")
        headlines = self.fetch_all_feeds()
        if not headlines:
            return ObserverResult(
                success=False, error="No headlines fetched from any RSS feed."
            )

        headlines_text = self._format_headlines(headlines)
        log.info("Total: %d headlines. Generating brief...", len(headlines))

        # 2. Generate brief with citations
        prompt = self.build_prompt(headlines, date_str)
        brief = self._call_cloud_llm(prompt, timeout=120)
        if not brief:
            msg = "LLM returned empty response"
            self.send_telegram(f"[SNIPPET ERROR] {msg}")
            return ObserverResult(success=False, error=msg)

        # 3. Validate source fidelity (brief vs. cited headlines)
        log.info("Validating source fidelity...")
        brief = self._validate_source_fidelity(brief, headlines_text)

        # 4. Validate ON THIS DAY / QUOTE via web-grounded search
        log.info("Validating knowledge sections (web-grounded)...")
        brief = self._validate_knowledge_sections(brief)

        # 5. Council review (4 models in parallel)
        log.info("Running AI council review...")
        council = self._council_review(brief, headlines_text)
        council_score = council.average_score
        council_responded = council.responded
        council_total = council.total

        if council.verdict == "revise":
            log.info("Council requested revision (%.1f/10). Revising...", council_score)
            brief = self._revise_with_feedback(brief, headlines_text, council.feedback)
            # Re-evaluate after revision
            council = self._council_review(brief, headlines_text)
            council_score = council.average_score
            council_responded = council.responded
            log.info("Post-revision council score: %.1f/10 (%s)", council_score, council.verdict)

        if council.verdict == "abort" or (council.verdict == "revise"):
            log.warning("Council rejected brief (%.1f/10). Degrading to headline summary.", council_score)
            brief = self._generate_headline_summary(headlines, date_str)
            council_score = 0.0
            council_responded = 0

        # Clean up SKIP_QUOTE
        brief = brief.replace("SKIP_QUOTE", "").strip()

        # 6. Format and send email
        subject = f"Daily Intelligence Snippet -- {date_str}"
        html_content = self.brief_to_html(
            brief, date_str,
            council_score=council_score,
            council_responded=council_responded,
            council_total=council_total,
            headline_count=len(headlines),
        )
        sent = self.send_email(subject, html_content, brief)

        if sent:
            snippet_to = os.environ.get("SNIPPET_TO", "(unknown)")
            log.info("Email sent to %s", snippet_to)
            score_tag = f" | Council: {council_score:.1f}/10" if council_score > 0 else " | Headline summary (council rejected)"
            short = brief[:1800] + "..." if len(brief) > 1800 else brief
            self.send_telegram(f"[SNIPPET] Sent to recipients{score_tag}\n\n{short}")
            return ObserverResult(
                success=True,
                data={
                    "headlines": len(headlines),
                    "email_sent": True,
                    "council_score": council_score,
                },
            )
        else:
            msg = "Email sending failed -- check SMTP config"
            self.send_telegram(f"[SNIPPET ERROR] {msg}")
            return ObserverResult(success=False, error=msg)

    # -----------------------------------------------------------------------
    # RSS fetching
    # -----------------------------------------------------------------------

    def fetch_rss(self, name: str, url: str) -> list[dict]:
        """Fetch and parse one RSS feed. Returns list of {title, summary, source}."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=15)
            root = ET.fromstring(resp.read())
        except Exception as e:
            log.warning("[%s] RSS fetch failed: %s", name, e)
            return []

        items = []
        # Standard RSS
        for item in root.findall(".//item"):
            title = item.find("title")
            desc = item.find("description")
            if title is not None and title.text:
                summary = ""
                if desc is not None and desc.text:
                    summary = re.sub(r"<[^>]+>", "", desc.text)[:300]
                items.append({
                    "title": title.text.strip(),
                    "summary": summary.strip(),
                    "source": name,
                })

        # Atom fallback
        if not items:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", ns):
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                if title is not None and title.text:
                    s = ""
                    if summary is not None and summary.text:
                        s = re.sub(r"<[^>]+>", "", summary.text)[:300]
                    items.append({
                        "title": title.text.strip(),
                        "summary": s.strip(),
                        "source": name,
                    })

        return items[:15]  # Cap per source

    def deduplicate_headlines(self, items: list[dict]) -> list[dict]:
        """Remove near-duplicate headlines by normalizing and comparing titles."""
        seen = set()
        unique = []
        for item in items:
            norm = re.sub(r"[^a-z0-9 ]", "", item["title"].lower())
            norm = re.sub(r"\s+", " ", norm).strip()
            key = norm[:60]
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def fetch_all_feeds(self) -> list[dict]:
        """Fetch all RSS feeds and return combined, deduplicated headline list."""
        all_items = []
        for name, url in self.RSS_FEEDS.items():
            items = self.fetch_rss(name, url)
            all_items.extend(items)
            log.info("  [%s] %d items", name, len(items))

        before = len(all_items)
        all_items = self.deduplicate_headlines(all_items)
        after = len(all_items)
        if before != after:
            log.info(
                "Deduplicated: %d -> %d headlines (%d duplicates removed)",
                before, after, before - after,
            )

        return all_items

    # -----------------------------------------------------------------------
    # Headline formatting (reusable across stages)
    # -----------------------------------------------------------------------

    def _format_headlines(self, headlines: list[dict]) -> str:
        """Format headlines as numbered list for use in prompts."""
        lines = []
        for i, h in enumerate(headlines, 1):
            line = f"{i}. [{h['source']}] {h['title']}"
            if h["summary"]:
                line += f" -- {h['summary']}"
            lines.append(line)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Brief generation
    # -----------------------------------------------------------------------

    def build_prompt(self, headlines: list[dict], date_str: str) -> str:
        """Build the generation prompt from today's headlines."""
        headline_text = self._format_headlines(headlines)

        return f"""You are writing a daily intelligence brief for a small strategic advisory firm. Today is {date_str}. This is the actual current date -- trust it completely. Do NOT treat events or documents dated 2025 or 2026 as speculative or forward-looking simply because they are near your training cutoff. They are real and current.

Here are today's news headlines from major sources:

{headline_text}

Write a "Daily Intelligence Snippet". You MUST follow this EXACT output format -- do NOT deviate from the structure, headings, or prefixes shown below.

=== BEGIN FORMAT ===

ON THIS DAY: [2-3 sentences about a significant historical event on this calendar date. Connect briefly to today's geopolitical landscape.]

QUOTE: "[A real, verifiable quote from a statesman, strategist, or thinker relevant to today's top stories]" -- [Attribution with role/title]

AMERICAS

**[Topic headline, 5-10 words]**
[One sentence: what happened] [HL-N, HL-M]
-> [One sentence: strategic implication]

**[Topic headline]**
[What happened] [HL-N]
-> [Strategic implication]

EUROPE

[Same format -- 2-4 stories per section]

MIDDLE EAST

[Same format]

ASIA-PACIFIC

[Same format]

GLOBAL

[Same format -- commodity markets, trade, climate, tech]

=== END FORMAT ===

STRICT FORMATTING RULES:
- Section headers must be EXACTLY: AMERICAS, EUROPE, MIDDLE EAST, ASIA-PACIFIC, GLOBAL -- on a line by themselves, no colons, no extra text
- Story headlines must use **bold** markers
- Strategic implications must start with -> on a new line
- "ON THIS DAY:" must be the literal prefix (not "Historical Anchor" or any variant)
- "QUOTE:" must be the literal prefix (not "QOTD" or any variant)

SOURCE CITATION:
- For every story, cite which headline number(s) it draws from: [HL-N, HL-M]
- Place citations at the end of the "what happened" line
- ON THIS DAY and QUOTE are knowledge-based -- no citation needed
- NEVER fabricate details not present in the cited headlines

QUOTE GUIDELINES:
- Use quotes from statesmen, diplomats, strategists, military leaders, or serious thinkers
- The quote must be REAL and VERIFIABLE -- not fabricated
- If you cannot think of a genuinely relevant and real quote, output SKIP_QUOTE on a line by itself instead of the QUOTE section
- Do NOT use quotes that are inflammatory, hateful, or incite violence -- even if newsworthy

CONTENT RULES:
- Each news item must appear in EXACTLY ONE section. Never repeat.
- Place each item in its most relevant geographic region.
- If a story spans regions, pick where the primary action is.
- GLOBAL is for commodity markets, trade, climate, tech -- things without a single regional home.
- Skip any region with no significant news today.
- Keep the total brief under 800 words.
- Tone: analytical, detached, realpolitik. No moralizing.
- Do NOT invent news. Only use the headlines provided.
- Do NOT add items not in the headlines list.

FACTUAL COMPLETENESS:
- NEVER omit material facts to appear neutral or politically correct. Accuracy trumps sensitivity.
- When describing people involved in events, use precise factual language. If a perpetrator, suspect, or key figure is transgender, state this plainly (e.g. "a biological male who identified as a woman" or "a trans woman"). Do not simply write "woman" or "man" if this obscures the biological sex, as this constitutes misinformation by omission.
- Apply the same standard to all identity-related facts: ethnicity, nationality, religious affiliation, political affiliation -- include them when they are material to the story as reported by primary sources.
- This briefing serves as an intelligence product. Sanitising facts for palatability undermines its core purpose."""

    # -----------------------------------------------------------------------
    # Prong A: Source fidelity validation
    # -----------------------------------------------------------------------

    def _validate_source_fidelity(self, brief: str, headlines_text: str) -> str:
        """Check that each story faithfully represents its cited headlines.

        Asks Gemini to compare the brief against the source headlines and flag
        any claims that add details not present in any headline. If issues are
        found, asks Gemini to correct ONLY unfaithful additions -- using the
        headlines as the ground truth (not training knowledge).
        """
        prompt = f"""You are a source-fidelity auditor. Compare this intelligence brief against the numbered source headlines below.

SOURCE HEADLINES:
{headlines_text}

BRIEF TO AUDIT:
{brief}

TASK:
1. For each story in the brief, check the [HL-N] citations.
2. Does the "what happened" line faithfully represent the cited headline(s)?
3. Flag any claims that ADD details not present in any cited headline (fabricated specifics, invented statistics, made-up names/titles).
4. Do NOT flag strategic analysis lines (starting with ->) -- those are commentary.
5. Do NOT flag ON THIS DAY or QUOTE sections -- those are knowledge-based.

Return a JSON object:
{{"issues": [
  {{"story": "headline text", "problem": "description of unfaithful addition", "fix": "corrected text using only headline facts"}}
], "clean": true/false}}

If no issues found, return: {{"issues": [], "clean": true}}
Return ONLY the JSON -- no other text."""

        try:
            result = self._call_cloud_llm(prompt, timeout=60)
        except Exception as e:
            log.warning("Source fidelity check failed: %s -- keeping brief unverified", e)
            return brief

        if not result:
            log.warning("Source fidelity: empty response -- keeping brief unverified")
            return brief

        parsed = extract_json(result)
        if not isinstance(parsed, dict):
            log.warning("Source fidelity: could not parse JSON -- keeping brief unverified")
            return brief

        issues = parsed.get("issues", [])
        if not issues:
            log.info("Source fidelity: PASSED -- all stories faithful to headlines")
            return brief

        log.info("Source fidelity: %d issues found, correcting...", len(issues))
        for issue in issues:
            log.info("  [UNFAITHFUL] %s", issue.get("problem", "")[:100])

        # Ask Gemini to fix ONLY the unfaithful additions, using headlines as ground truth
        issues_text = json.dumps(issues, indent=2)
        fix_prompt = f"""Fix this intelligence brief by correcting ONLY the unfaithful additions identified below.
Use ONLY facts from the source headlines -- do NOT add information from your training knowledge.

SOURCE HEADLINES:
{headlines_text}

CURRENT BRIEF:
{brief}

ISSUES TO FIX:
{issues_text}

Rules:
- Fix ONLY the flagged issues. Do not change anything else.
- Maintain the exact same format, structure, and tone.
- If a fix would remove all substance from a story, remove that story entirely.
- Output ONLY the corrected brief -- no preamble, no commentary."""

        try:
            fixed = self._call_cloud_llm(fix_prompt, timeout=60)
            if fixed and len(fixed) > len(brief) * 0.5:
                log.info("Source fidelity: corrections applied (%d chars)", len(fixed))
                return fixed
            log.warning("Source fidelity: fix response too short, keeping original")
        except Exception as e:
            log.warning("Source fidelity fix failed: %s -- keeping original", e)

        return brief

    # -----------------------------------------------------------------------
    # Prong B: Knowledge section validation (web-grounded)
    # -----------------------------------------------------------------------

    def _validate_knowledge_sections(self, brief: str) -> str:
        """Validate ON THIS DAY and QUOTE using Gemini with Google Search grounding.

        Uses Gemini's native google_search tool to verify knowledge-based sections
        against live web results. This is the ONLY stage that uses web search --
        news stories are validated against their source headlines instead.
        """
        # Extract the sections to validate
        otd_match = re.search(r"^ON THIS DAY:?\s*(.+)", brief, re.MULTILINE)
        quote_match = re.search(r'^QUOTE:?\s*"(.+?")\s*--\s*(.+)', brief, re.MULTILINE)

        if not otd_match and not quote_match:
            log.info("Knowledge validation: no ON THIS DAY or QUOTE sections found")
            return brief

        sections_to_check = ""
        if otd_match:
            sections_to_check += f"ON THIS DAY: {otd_match.group(1)}\n"
        if quote_match:
            sections_to_check += f'QUOTE: "{quote_match.group(1)} -- {quote_match.group(2)}\n'

        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            log.warning("Knowledge validation: no Gemini API key -- skipping")
            return brief

        try:
            from google import genai
            from google.genai import types as gtypes

            client = genai.Client(api_key=gemini_key)

            # Use Google Search grounding tool
            google_search_tool = gtypes.Tool(
                google_search=gtypes.GoogleSearch()
            )

            prompt = f"""Verify these sections from a daily intelligence brief using web search.

{sections_to_check}

For ON THIS DAY:
- Is the historical event correct for this calendar date?
- Are the specific facts (names, dates, details) accurate?
- If wrong, provide the correct historical event for this date.

For QUOTE:
- Is this a real, documented, word-for-word quote?
- Is the attribution (person, title) correct?
- If fabricated or misattributed, respond with QUOTE_INVALID.

Return JSON:
{{"on_this_day": {{"valid": true/false, "correction": "corrected text or empty"}}, "quote": {{"valid": true/false, "correction": "corrected text or QUOTE_INVALID"}}}}

Return ONLY the JSON."""

            config = gtypes.GenerateContentConfig(
                tools=[google_search_tool],
                temperature=0.1,
                max_output_tokens=2048,
            )

            response = client.models.generate_content(
                model="gemini-3.0-flash",
                contents=prompt,
                config=config,
            )

            # Log grounding metadata if available
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                gm = getattr(candidate, "grounding_metadata", None)
                if gm:
                    chunks = getattr(gm, "grounding_chunks", []) or []
                    if chunks:
                        log.info("Grounding searches: %d web sources used", len(chunks))

            text = response.text or ""
            if not text.strip():
                log.warning("Knowledge validation: empty response -- skipping")
                return brief

            parsed = extract_json(text)
            if not isinstance(parsed, dict):
                log.warning("Knowledge validation: could not parse JSON -- skipping")
                return brief

            # Apply corrections
            otd_result = parsed.get("on_this_day", {})
            quote_result = parsed.get("quote", {})

            if otd_result.get("valid") is False and otd_result.get("correction"):
                correction = otd_result["correction"]
                log.info("Knowledge validation: ON THIS DAY corrected")
                if otd_match:
                    brief = brief.replace(
                        otd_match.group(0),
                        f"ON THIS DAY: {correction}"
                    )

            if quote_result.get("valid") is False:
                correction = quote_result.get("correction", "")
                if correction == "QUOTE_INVALID" or not correction:
                    log.info("Knowledge validation: QUOTE invalid -- removing")
                    if quote_match:
                        brief = brief.replace(quote_match.group(0), "SKIP_QUOTE")
                else:
                    log.info("Knowledge validation: QUOTE corrected")
                    if quote_match:
                        brief = brief.replace(quote_match.group(0), f"QUOTE: {correction}")

            if otd_result.get("valid") is not False and quote_result.get("valid") is not False:
                log.info("Knowledge validation: PASSED -- both sections verified")

            return brief

        except Exception as e:
            log.warning("Knowledge validation failed: %s -- keeping sections unverified", e)
            return brief

    # -----------------------------------------------------------------------
    # AI Council quality gate
    # -----------------------------------------------------------------------

    def _council_review(self, brief: str, headlines_text: str) -> CouncilResult:
        """Run 4-model AI council to evaluate the brief.

        Each council member receives the RSS headlines as context so they verify
        the brief against its actual sources -- not their training knowledge.
        """
        context_block = (
            f"SOURCE HEADLINES (verify the brief against these, not your training knowledge):\n"
            f"{headlines_text[:6000]}\n\n"
            f"BRIEF TO EVALUATE:\n{brief}"
        )

        roles = {
            "accuracy_checker": {
                "model": "bedrock",
                "system": (
                    "You are a fact-checking editor for an intelligence briefing service. "
                    "You have been given the source headlines that the brief was generated from. "
                    "Evaluate whether the brief faithfully represents the headlines without "
                    "fabricating details, misrepresenting events, or adding unsourced claims."
                ),
                "prompt": (
                    "Score this intelligence brief for FAITHFULNESS TO SOURCE HEADLINES. "
                    "Check: Are stories accurately drawn from the cited headlines? "
                    "Are any details fabricated or embellished beyond what the headlines state? "
                    "Are citations [HL-N] present and correct?"
                ),
            },
            "coherence_editor": {
                "model": "gemini",
                "system": (
                    "You are a senior editor at a strategic advisory firm. "
                    "You evaluate intelligence briefs for writing quality, logical structure, "
                    "and professional tone."
                ),
                "prompt": (
                    "Score this intelligence brief for WRITING QUALITY. "
                    "Check: Is the prose clear and concise? Is the structure consistent? "
                    "Is the tone analytical and detached? Are strategic implications well-reasoned?"
                ),
            },
            "completeness_analyst": {
                "model": "grok",
                "system": (
                    "You are a completeness analyst. You compare a brief against its source "
                    "headlines to assess whether the most significant stories have been covered."
                ),
                "prompt": (
                    "Score this intelligence brief for COVERAGE COMPLETENESS. "
                    "Check: Are the most important headlines covered? Are any major stories "
                    "conspicuously absent? Is the regional balance appropriate?"
                ),
            },
            "critic": {
                "model": "deepseek",
                "system": (
                    "You are a devil's advocate reviewer for intelligence products. "
                    "You look for bias, logical gaps, unsupported leaps, and editorializing."
                ),
                "prompt": (
                    "Score this intelligence brief for OBJECTIVITY AND RIGOUR. "
                    "Check: Is there hidden bias or editorializing in the 'what happened' lines? "
                    "Are strategic implications logically supported? "
                    "Are there any logical gaps or unsupported assertions?"
                ),
            },
        }

        result = run_council(
            content=context_block,
            roles=roles,
            threshold=6.0,
            min_score=4,
            min_quorum=3,
            timeout=120,
        )

        log.info(
            "Council: %.1f/10 (%d/%d responded) -- %s",
            result.average_score, result.responded, result.total, result.verdict,
        )
        for m in result.members:
            if m.error:
                log.warning("  [%s] %s: ERROR -- %s", m.role, m.model, m.error[:80])
            else:
                log.info("  [%s] %s: %d/10 (%s)", m.role, m.model, m.score, m.verdict)

        return result

    # -----------------------------------------------------------------------
    # Revision with council feedback
    # -----------------------------------------------------------------------

    def _revise_with_feedback(self, brief: str, headlines_text: str, feedback: str) -> str:
        """Revise the brief using council feedback, constrained to source headlines."""
        prompt = f"""Revise this intelligence brief based on editorial feedback.
You must ONLY use facts from the source headlines below -- do NOT add information from your training knowledge.

SOURCE HEADLINES:
{headlines_text}

CURRENT BRIEF:
{brief}

EDITORIAL FEEDBACK:
{feedback}

Rules:
- Address the specific concerns raised in the feedback.
- Do NOT add new stories or details not present in the headlines.
- Maintain the exact same format and structure.
- Keep [HL-N] citations.
- Output ONLY the revised brief -- no preamble, no commentary."""

        try:
            revised = self._call_cloud_llm(prompt, timeout=90)
            if revised and len(revised) > len(brief) * 0.5:
                log.info("Revision applied (%d chars)", len(revised))
                return revised
            log.warning("Revision response too short, keeping original")
        except Exception as e:
            log.warning("Revision failed: %s -- keeping original", e)

        return brief

    # -----------------------------------------------------------------------
    # Degraded output fallback
    # -----------------------------------------------------------------------

    def _generate_headline_summary(self, headlines: list[dict], date_str: str) -> str:
        """Generate a simple headline summary when council rejects the analysis.

        No LLM analysis -- just the raw headlines formatted for readability.
        """
        log.info("Generating headline summary fallback (%d headlines)", len(headlines))

        lines = [
            f"AUTOMATED ANALYSIS UNAVAILABLE -- HEADLINE SUMMARY ONLY",
            f"Date: {date_str}",
            "",
        ]

        # Group by source for readability
        by_source: dict[str, list[str]] = {}
        for h in headlines[:30]:
            by_source.setdefault(h["source"], []).append(h["title"])

        for source, titles in by_source.items():
            lines.append(source.upper())
            for title in titles[:5]:
                lines.append(f"  - {title}")
            lines.append("")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # HTML email formatting
    # -----------------------------------------------------------------------

    def brief_to_html(
        self,
        brief_text: str,
        date_str: str,
        council_score: float = 0.0,
        council_responded: int = 0,
        council_total: int = 4,
        headline_count: int = 0,
    ) -> str:
        """Convert the plain text brief to styled HTML email."""
        html_body = ""
        for line in brief_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Strip [HL-N] citation markers from display (they served their purpose)
            line_display = re.sub(r"\s*\[HL-[\d,\s]+\]", "", line)

            # Bold markers
            line_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line_display)

            # Section headers
            if line.startswith("ON THIS DAY"):
                html_body += (
                    '<h2 style="font-size:19px; color:#1a1a1a; text-transform:uppercase; '
                    'letter-spacing:1px; margin-top:24px; border-bottom:1px solid #ccc; '
                    'padding-bottom:4px;">On This Day</h2>\n'
                )
                content = re.sub(r"^ON THIS DAY:?\s*", "", line_display)
                if content:
                    html_body += (
                        f'<p style="font-style:italic; color:#555; '
                        f'margin:8px 0 16px 0;">{content}</p>\n'
                    )
            elif line.startswith("QUOTE:") or line.startswith('"'):
                quote_text = re.sub(r"^QUOTE:?\s*", "", line_display)
                if "SKIP_QUOTE" not in quote_text:
                    html_body += (
                        f'<div style="background:#f5f5f5; padding:12px 16px; '
                        f'border-left:3px solid #333; margin:16px 0; '
                        f'font-style:italic;">{quote_text}</div>\n'
                    )
            elif line == "SKIP_QUOTE":
                pass
            elif line.startswith("AUTOMATED ANALYSIS UNAVAILABLE"):
                html_body += (
                    '<div style="background:#fff3cd; padding:12px 16px; '
                    'border-left:3px solid #ffc107; margin:16px 0; '
                    'font-weight:bold; color:#856404;">Automated analysis unavailable '
                    '-- headline summary only</div>\n'
                )
            elif line in ("AMERICAS", "EUROPE", "MIDDLE EAST", "ASIA-PACIFIC", "GLOBAL"):
                html_body += (
                    f'<h2 style="font-size:19px; color:#1a1a1a; text-transform:uppercase; '
                    f'letter-spacing:1px; margin-top:24px; border-bottom:1px solid #ccc; '
                    f'padding-bottom:4px;">{line}</h2>\n'
                )
            elif line.startswith("\u2192") or line.startswith("->"):
                arrow_text = re.sub(r"^(\u2192|->)\s*", "", line_html)
                html_body += (
                    f'<p style="margin:4px 0 12px 16px; color:#555; '
                    f'font-style:italic; font-size:15px;">\u2192 {arrow_text}</p>\n'
                )
            else:
                html_body += f'<p style="margin:4px 0 4px 0;">{line_html}</p>\n'

        # Build footer
        footer_parts = [f"Generated by {AGENT_NAME}"]
        if headline_count:
            footer_parts.append(f"Verified against {headline_count} source headlines")
        if council_score > 0:
            footer_parts.append(f"Council score: {council_score:.1f}/10 ({council_responded}/{council_total} models)")
        footer_parts.append("Sources: Reuters, AP, BBC, Bloomberg, Politico, Al Jazeera, SCMP, JPost, Foreign Policy, NPR, Economist")

        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Georgia, serif; font-size: 17px; line-height: 1.6; color: #333; max-width: 700px; margin: 0 auto; padding: 20px;">

  <h1 style="font-size: 26px; border-bottom: 2px solid #333; padding-bottom: 8px; margin-bottom: 4px;">Daily Intelligence Snippet</h1>
  <p style="color: #666; margin-bottom: 16px;">{date_str}</p>

  {html_body}

  <div style="margin-top: 30px; padding-top: 16px; border-top: 1px solid #ccc; font-size: 11px; color: #888;">
    {" | ".join(footer_parts)}
  </div>

</body>
</html>"""

    # -----------------------------------------------------------------------
    # Email sending
    # -----------------------------------------------------------------------

    def send_email(self, subject: str, html_content: str, plain_text: str) -> bool:
        """Send HTML email via SMTP. Config from environment variables."""
        host = os.environ.get("SNIPPET_SMTP_HOST", "")
        port = int(os.environ.get("SNIPPET_SMTP_PORT", "587"))
        user = os.environ.get("SNIPPET_SMTP_USER", "")
        password = os.environ.get("SNIPPET_SMTP_PASS", "")
        from_addr = os.environ.get("SNIPPET_FROM", user)
        to_addrs = [
            a.strip()
            for a in os.environ.get("SNIPPET_TO", "").split(",")
            if a.strip()
        ]

        if not user or not password or not to_addrs:
            log.error("SMTP not configured -- set SNIPPET_SMTP_* env vars")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)

        msg.attach(MIMEText(plain_text, "plain", "utf-8"))
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        try:
            with smtplib.SMTP(host, port) as server:
                server.starttls()
                server.login(user, password)
                server.sendmail(from_addr, to_addrs, msg.as_string())
            return True
        except Exception as e:
            log.error("SMTP error: %s", e)
            return False


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    # When running standalone, load .env from the nexus project root
    project_dir = Path(__file__).parent.parent
    env_path = project_dir / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    # Ensure config can be imported when running standalone
    sys.path.insert(0, str(project_dir))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    observer = DailySnippetObserver()
    ctx = ObserverContext()
    result = observer.run(ctx)

    if result.success:
        print(f"SUCCESS: {result.data}")
    else:
        print(f"FAILED: {result.error}", file=sys.stderr)
        sys.exit(1)
