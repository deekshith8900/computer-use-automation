"""
agent/browser.py — Playwright async browser wrapper.

Provides:
  - Locator resolution with priority: aria-label > data-testid > role > text > css
  - Fallback chain (try primary, then fallbacks)
  - Screenshot capture to evidence/
  - DOM snapshot on failure
  - Accessibility tree text extraction (for LLM perception)
  - Checkpoint evaluation
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Locator as PlaywrightLocator,
    TimeoutError as PlaywrightTimeoutError,
)

from agent.artifact import Locator, Checkpoint


# ─── Locator Resolution ───────────────────────────────────────────────────────

async def resolve_locator(page: Page, loc: Locator, timeout: int = 5000) -> PlaywrightLocator:
    """
    Resolve an artifact Locator to a Playwright locator.
    Priority: aria-label > data-testid > role > text > css > xpath

    Tries primary strategy, then falls back to fallback locators in order.
    Raises PlaywrightTimeoutError if none match.
    """
    candidates = [loc.to_dict()] + loc.fallbacks
    last_error = None
    for candidate in candidates:
        try:
            pl = _make_locator(page, candidate)
            await pl.first.wait_for(state="attached", timeout=timeout)
            return pl
        except Exception as e:
            last_error = e
            continue
    raise PlaywrightTimeoutError(f"No locator resolved for {loc.to_dict()}. Last error: {last_error}")


def _make_locator(page: Page, loc_dict: dict) -> PlaywrightLocator:
    strategy = loc_dict["strategy"]
    value = loc_dict["value"]
    name = loc_dict.get("name")

    if strategy == "aria-label":
        return page.locator(f'[aria-label="{value}"]')
    elif strategy == "data-testid":
        return page.locator(f'[data-testid="{value}"]')
    elif strategy == "role":
        kwargs = {}
        if name:
            kwargs["name"] = name
        return page.get_by_role(value, **kwargs)
    elif strategy == "text":
        return page.get_by_text(value, exact=False)
    elif strategy == "css":
        return page.locator(value)
    elif strategy == "xpath":
        return page.locator(f"xpath={value}")
    else:
        raise ValueError(f"Unknown locator strategy: {strategy}")


# ─── Checkpoint Evaluation ────────────────────────────────────────────────────

async def evaluate_checkpoint(page: Page, checkpoint: Checkpoint, timeout: int = 5000) -> tuple[bool, str]:
    """
    Evaluate a checkpoint assertion.
    Returns (passed: bool, detail: str).
    """
    try:
        if checkpoint.type == "element_visible":
            pl = await resolve_locator(page, checkpoint.locator, timeout)
            await pl.first.wait_for(state="visible", timeout=timeout)
            return True, "element visible"

        elif checkpoint.type == "element_not_found":
            pl = _make_locator(page, checkpoint.locator.to_dict())
            count = await pl.count()
            if count == 0:
                return True, "element not present"
            return False, f"element still present (count={count})"

        elif checkpoint.type == "url_contains":
            url = page.url
            if checkpoint.value in url:
                return True, f"URL contains '{checkpoint.value}'"
            return False, f"URL '{url}' does not contain '{checkpoint.value}'"

        elif checkpoint.type == "text_contains":
            if checkpoint.locator:
                pl = await resolve_locator(page, checkpoint.locator, timeout)
                text = await pl.first.inner_text()
            else:
                text = await page.inner_text("body")
            if checkpoint.value in text:
                return True, f"text contains '{checkpoint.value}'"
            return False, f"text does not contain '{checkpoint.value}'"

        elif checkpoint.type == "element_count":
            pl = _make_locator(page, checkpoint.locator.to_dict())
            count = await pl.count()
            if count == checkpoint.count:
                return True, f"element count is {count}"
            return False, f"expected {checkpoint.count} elements, found {count}"

        else:
            return False, f"unknown checkpoint type: {checkpoint.type}"

    except Exception as e:
        return False, str(e)


# ─── Evidence Capture ─────────────────────────────────────────────────────────

async def take_screenshot(page: Page, evidence_dir: str, label: str) -> str:
    """Take a screenshot and save to evidence_dir. Returns path."""
    directory = Path(evidence_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%H%M%S")
    filename = f"{label}_{timestamp}.png"
    path = directory / filename
    await page.screenshot(path=str(path), full_page=True)
    return str(path)


async def get_dom_snapshot(page: Page) -> str:
    """Return a truncated HTML snapshot of the current page for debugging."""
    html = await page.content()
    return html[:8000] + ("..." if len(html) > 8000 else "")


async def get_accessibility_text(page: Page) -> str:
    """
    Extract a text representation of the page's accessibility tree.
    Used as the primary perception input for the LLM during discovery.
    """
    try:
        snapshot = await page.accessibility.snapshot()
        if not snapshot:
            return await page.inner_text("body")
        return _flatten_ax(snapshot)
    except Exception:
        return await page.inner_text("body")


def _flatten_ax(node: dict, depth: int = 0, max_depth: int = 8) -> str:
    """Recursively flatten an accessibility tree node to text."""
    if depth > max_depth:
        return ""
    parts = []
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    line = "  " * depth
    if role and role not in ("none", "presentation", "generic"):
        line += f"[{role}]"
        if name:
            line += f' "{name}"'
        if value:
            line += f" = {value}"
        parts.append(line)

    for child in node.get("children", []):
        parts.append(_flatten_ax(child, depth + 1, max_depth))
    return "\n".join(p for p in parts if p.strip())


# ─── Browser Manager ─────────────────────────────────────────────────────────

class BrowserManager:
    """
    Manages a single Playwright browser/context/page lifecycle.
    Supports pause/resume for HITL handoff.
    """

    def __init__(self, headless: bool = False, evidence_dir: str = "evidence"):
        self.headless = headless
        self.evidence_dir = evidence_dir
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def start(self) -> Page:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        self._page = await self._context.new_page()
        return self._page

    @property
    def page(self) -> Page:
        if not self._page:
            raise RuntimeError("Browser not started. Call await browser_manager.start() first.")
        return self._page

    async def screenshot(self, label: str) -> str:
        return await take_screenshot(self._page, self.evidence_dir, label)

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> None:
        await self._page.goto(url, wait_until=wait_until, timeout=15000)

    async def fill(self, locator: Locator, value: str) -> None:
        pl = await resolve_locator(self._page, locator)
        await pl.first.fill(value)

    async def click(self, locator: Locator) -> None:
        pl = await resolve_locator(self._page, locator)
        await pl.first.click()

    async def select_option(self, locator: Locator, value: str) -> None:
        pl = await resolve_locator(self._page, locator)
        await pl.first.select_option(value)

    async def extract_text(self, locator: Locator) -> str:
        pl = await resolve_locator(self._page, locator)
        return (await pl.first.inner_text()).strip()

    async def wait(self, ms: int) -> None:
        await self._page.wait_for_timeout(ms)

    async def check_business_outcome(self, signal: str) -> bool:
        """Check if the current page contains a known business outcome signal."""
        try:
            body = await self._page.inner_text("body")
            return signal.lower() in body.lower()
        except Exception:
            return False

    async def get_page_text(self) -> str:
        return await get_accessibility_text(self._page)

    async def get_screenshot_b64(self, scale: float = 0.5) -> str:
        """Return current page screenshot as base64 (for LLM vision input).

        scale=0.5 reduces image to 50% size, cutting token usage ~75%
        while keeping text readable for the model.
        """
        data = await self._page.screenshot(
            full_page=False,
            scale="css",  # use CSS pixels (device-independent)
        )
        # Resize using PIL if available, else use raw screenshot
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            data = buf.getvalue()
        except ImportError:
            pass  # PIL not installed — use original screenshot
        return base64.b64encode(data).decode()
