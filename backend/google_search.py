import os
import json
import asyncio
import aiohttp
import re
import csv
import sys
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langdetect import detect, DetectorFactory
import time
import random
import logging
from urllib.parse import quote_plus, unquote, parse_qs, urlunparse
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

from serp_captcha_recovery import (
    SerpBlockedError,
    SerpCaptchaError,
    default_serp_browser,
    session_dir_for,
)

# Load environment variables
load_dotenv()

# Ensures language detection is consistent
DetectorFactory.seed = 0

# Configure logging for Google scraper
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("google_search")


def has_display_server() -> bool:
    """Return True when a headed Linux browser has a display to attach to."""
    if sys.platform.startswith("linux"):
        return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))
    return True


def resolve_browser_headless(requested_headless: bool, logger=None) -> bool:
    """Force headless mode when headful Chromium cannot run in this environment."""
    if requested_headless:
        return True

    if has_display_server():
        return False

    message = (
        "Headful browser requested, but no DISPLAY/WAYLAND_DISPLAY is available. "
        "Falling back to headless mode. Use xvfb-run or run on a desktop session for headful mode."
    )
    if logger is not None:
        logger.warning(message)
    else:
        print(message)
    return True


BASE_CHROMIUM_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--no-sandbox",
]

# ============================================================================
# GOOGLE SCRAPER CONFIGURATION (from google_version_2.py)
# ============================================================================
MAX_K        = 100
GOOGLE_URL   = os.getenv('GOOGLE_URL', 'https://www.google.com/search')
CHUNK_SIZE   = 10
SESSION_PATH = session_dir_for(default_serp_browser())

TRACKING_PARAMS = {
    "srsltid", "utm_source", "utm_medium", "utm_campaign", "utm_term",
    "utm_content", "ref", "referrer", "fbclid", "gclid", "msclkid",
    "_ga", "mc_cid", "mc_eid",
}

BLOCKED_HOSTS = {
    "google.com", "webcache.googleusercontent.com", "support.google.com",
    "policies.google.com", "accounts.google.com", "translate.google.com", "play.google.com",
}

EXTRACT_JS = """
() => {
    const EXCLUDED = [
        '#kp-wp-tab-overview', '[data-initq]', '.related-question-pair',
        '.ULSxyf', '[data-text-ad]', '.commercial-unit-desktop-top',
        '.commercial-unit-desktop-rhs', '.g-blk', '#tads', '#bottomads',
        '[data-text-ad]', '.ads-ad', '#bottomads', '#tads'
    ];

    function resolveHref(raw) {
        if (!raw) return null;
        if (raw.startsWith('/url?q=') || raw.includes('google.com/url?q=')) {
            try {
                const u = new URL(raw.startsWith('/') ? 'https://google.com' + raw : raw);
                const q = u.searchParams.get('q');
                if (q && q.startsWith('http')) return decodeURIComponent(q);
            } catch(e) {}
        }
        if (raw.startsWith('http') && !raw.includes('google.com')) return raw;
        return null;
    }

    function isExcluded(el) {
        return EXCLUDED.some(sel => {
            try { return el.closest(sel) !== null; } catch(e) { return false; }
        });
    }

    const urls = [];
    const searchRoot =
        document.querySelector('div#search')   ||
        document.querySelector('div#rso')      ||
        document.querySelector('div#main')     ||
        document.querySelector('div#cnt');
    
    if (searchRoot) {
        const headings = Array.from(searchRoot.querySelectorAll(
            'h3, div[role="heading"], span[role="heading"]'
        ));

        for (const heading of headings) {
            if (isExcluded(heading)) continue;
            let el = heading.closest('a[href]');
            if (!el) {
                let p = heading.parentElement;
                let depth = 0;
                while (p && depth < 10) {
                    const anchors = Array.from(p.querySelectorAll('a[href]'));
                    const match = anchors.find(a => {
                        const h = a.getAttribute('href') || '';
                        return (h.startsWith('http') && !h.includes('google.com')) || h.startsWith('/url?q=');
                    });
                    if (match) { el = match; break; }
                    p = p.parentElement;
                    depth++;
                }
            }
            if (el) {
                const href = resolveHref(el.getAttribute('href'));
                if (href && !urls.includes(href)) urls.push(href);
            }
        }
    }

    // Fallback for mobile layouts without explicit headings
    if (urls.length === 0) {
        const allAnchors = Array.from(document.querySelectorAll('a[href]'));
        for (const a of allAnchors) {
            if (isExcluded(a)) continue;
            const href = resolveHref(a.getAttribute('href') || '');
            if (!href) continue;
            const text = (a.textContent || '').trim();
            if (text.length < 5) continue;
            if (!urls.includes(href)) urls.push(href);
        }
    }
    return urls;
}
"""

# ============================================================================
# CONFIGURATION
# ============================================================================
PROXY_URL = os.getenv('GOOGLE_PROXY_URL', 'http://108.59.14.203:13080')

# ============================================================================
# LOAD DR DATA FROM CSV
# ============================================================================

def load_dr_data(csv_path="Inputs/Data .csv"):
    """Load Domain Rating data from CSV into a dictionary"""
    dr_dict = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    domain = row[0].strip()
                    try:
                        dr = float(row[1].strip())
                        dr_dict[domain] = dr
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. All domains will get authority=5")
    return dr_dict

DR_DATA = load_dr_data()

# ============================================================================
# GOOGLE SCRAPER HELPER FUNCTIONS (from google_version_2.py)
# ============================================================================

def normalize_url(href: str) -> str:
    """Strip fragment, tracking params, and trailing slash."""
    try:
        p = urlparse(href)
        qs_clean = "&".join(
            f"{k}={v}"
            for part in (p.query or "").split("&")
            for k, _, v in [part.partition("=")]
            if k and k not in TRACKING_PARAMS
        )
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), p.params, qs_clean, ""))
    except Exception:
        return href

def is_organic_host(href: str) -> bool:
    """Check if host is a legitimate search result."""
    if not href or not href.startswith("http"):
        return False
    try:
        host = urlparse(href).netloc.lower().lstrip("www.")
        return not any(host == b or host.endswith("." + b) for b in BLOCKED_HOSTS)
    except Exception:
        return False

def get_hardened_fingerprint(device_type="desktop", serp_browser="chromium"):
    """Generates localized and device-specific fingerprints."""
    browser = (serp_browser or "chromium").lower()
    if device_type == "mobile":
        if browser == "firefox":
            user_agent = (
                "Mozilla/5.0 (Android 10; Mobile; rv:122.0) "
                "Gecko/122.0 Firefox/122.0"
            )
        else:
            user_agent = (
                "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36"
            )
        viewport = {"width": 412, "height": 915}
    else:
        if browser == "firefox":
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) "
                "Gecko/20100101 Firefox/131.0"
            )
        else:
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
        viewport = {"width": 1280, "height": 800}

    return {
        "user_agent": user_agent,
        "viewport": viewport,
        "hardware_concurrency": random.choice([4, 8, 16]),
        "device_memory": random.choice([4, 8]),
    }


def _build_proxy_cfg(use_proxy: bool, proxy_url: str | None):
    if proxy_url:
        return {"server": proxy_url}
    if use_proxy:
        return {
            "server": os.getenv("OXYLABS_PROXY_SERVER", "http://disp.oxylabs.io:8007"),
            "username": os.getenv("OXYLABS_PROXY_USERNAME", "user-LMX_P_N0PoY"),
            "password": os.getenv("OXYLABS_PROXY_PASSWORD", "5_vQlWT3~XfrM6S"),
        }
    return None


def _launch_serp_context_sync(p, serp_browser, user_data_dir, headless, proxy_cfg, fingerprint):
    common = dict(
        user_data_dir=user_data_dir,
        headless=headless,
        proxy=proxy_cfg,
        user_agent=fingerprint["user_agent"],
        viewport=fingerprint["viewport"],
    )
    if serp_browser == "firefox":
        return p.firefox.launch_persistent_context(**common)
    return p.chromium.launch_persistent_context(
        **common,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )


async def _launch_serp_context_async(p, serp_browser, user_data_dir, headless, proxy_cfg, fingerprint):
    common = dict(
        user_data_dir=user_data_dir,
        headless=headless,
        proxy=proxy_cfg,
        user_agent=fingerprint["user_agent"],
        viewport=fingerprint["viewport"],
    )
    if serp_browser == "firefox":
        return await p.firefox.launch_persistent_context(**common)
    return await p.chromium.launch_persistent_context(
        **common,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
    )


SERP_PAGE_PROBE_JS = """
() => {
    const title = document.title || '';
    const bodyText = (document.body && document.body.innerText) ? document.body.innerText.slice(0, 4000) : '';
    const hasSearchRoot = !!(
        document.querySelector('div#search') ||
        document.querySelector('div#rso') ||
        document.querySelector('div#main')
    );
    const lowerTitle = title.toLowerCase();
    const lowerBody = bodyText.toLowerCase();
    const blockPhrases = [
        'unusual traffic',
        'not a robot',
        'before you continue',
        'verify you are human',
        'captcha',
    ];
    const blockedText = blockPhrases.some(p => lowerTitle.includes(p) || lowerBody.includes(p));
    return { title, hasSearchRoot, blockedText, bodySnippet: bodyText.slice(0, 200) };
}
"""


def _manual_captcha_wait_seconds() -> int:
    try:
        return max(0, int(os.getenv("SERP_CAPTCHA_MANUAL_WAIT_SEC", "120")))
    except ValueError:
        return 120


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _classify_probe(probe: dict, page_url: str, serp_browser: str, link_count: int) -> None:
    if "/sorry/" in (page_url or ""):
        log.warning("CAPTCHA hit. Solve it in the browser window.")
        raise SerpCaptchaError(browser=serp_browser, reason="sorry")

    if probe.get("blockedText"):
        log.warning("SERP block detected (title/body): %s", probe.get("title"))
        raise SerpCaptchaError(
            browser=serp_browser,
            message="SERP block page detected",
            reason="block_text",
        )

    if link_count == 0 and not probe.get("hasSearchRoot"):
        log.warning(
            "Soft SERP block: no organic links and no search root (title=%r)",
            probe.get("title"),
        )
        raise SerpBlockedError(
            browser=serp_browser,
            message="Empty SERP with no search results container",
        )


def _log_serp_diagnostics(page_url: str, probe: dict, link_count: int) -> None:
    log.info(
        "SERP diagnostics: url=%s title=%r links=%d hasSearchRoot=%s",
        page_url,
        probe.get("title"),
        link_count,
        probe.get("hasSearchRoot"),
    )


def _check_captcha_sync(page, serp_browser: str, raw_hrefs: list) -> None:
    probe = page.evaluate(SERP_PAGE_PROBE_JS)
    _classify_probe(probe, page.url, serp_browser, len(raw_hrefs or []))
    if len(raw_hrefs or []) == 0:
        _log_serp_diagnostics(page.url, probe, 0)


async def _check_captcha_async(page, serp_browser: str, raw_hrefs: list) -> None:
    probe = await page.evaluate(SERP_PAGE_PROBE_JS)
    _classify_probe(probe, page.url, serp_browser, len(raw_hrefs or []))
    if len(raw_hrefs or []) == 0:
        _log_serp_diagnostics(page.url, probe, 0)


def _wait_for_manual_captcha_solve_sync(page, serp_browser: str) -> None:
    if not _env_truthy("SURFOX_HEADFUL"):
        return
    wait_sec = _manual_captcha_wait_seconds()
    if wait_sec <= 0 or "/sorry/" not in (page.url or ""):
        return
    log.warning("Waiting up to %ds for manual CAPTCHA solve...", wait_sec)
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if "/sorry/" not in (page.url or ""):
            log.info("CAPTCHA cleared manually.")
            return
        time.sleep(2)
    raise SerpCaptchaError(browser=serp_browser, reason="sorry", message="CAPTCHA not solved in time")


async def _wait_for_manual_captcha_solve_async(page, serp_browser: str) -> None:
    if not _env_truthy("SURFOX_HEADFUL"):
        return
    wait_sec = _manual_captcha_wait_seconds()
    if wait_sec <= 0 or "/sorry/" not in (page.url or ""):
        return
    log.warning("Waiting up to %ds for manual CAPTCHA solve...", wait_sec)
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if "/sorry/" not in (page.url or ""):
            log.info("CAPTCHA cleared manually.")
            return
        await asyncio.sleep(2)
    raise SerpCaptchaError(browser=serp_browser, reason="sorry", message="CAPTCHA not solved in time")


async def apply_stealth(context, fingerprint):
    """Hide automation indicators."""
    await context.add_init_script(f"""
        Object.defineProperty(navigator, 'webdriver', {{get: () => undefined}});
        Object.defineProperty(navigator, 'hardwareConcurrency', {{get: () => {fingerprint['hardware_concurrency']} }});
        Object.defineProperty(navigator, 'deviceMemory', {{get: () => {fingerprint['device_memory']} }});
    """)

def scrape_google_results(keyword, k=20, headless=False, use_proxy=False, device="desktop"):
    """Scrape Google search results using Playwright (sync wrapper)."""
    return asyncio.run(_scrape_google_results_async(keyword, k, headless, use_proxy, device))

def _scrape_google_results_sync(
    keyword,
    k=20,
    headless=False,
    use_proxy=False,
    device="desktop",
    serp_browser=None,
    proxy_url=None,
):
    """Sync Playwright fallback for Windows selector loop environments."""
    serp_browser = serp_browser or default_serp_browser()
    fingerprint = get_hardened_fingerprint(device, serp_browser)
    proxy_cfg = _build_proxy_cfg(use_proxy, proxy_url)
    user_data_dir = session_dir_for(serp_browser)

    urls = []
    with sync_playwright() as p:
        try:
            context = _launch_serp_context_sync(
                p, serp_browser, user_data_dir, headless, proxy_cfg, fingerprint
            )
        except Exception as exc:
            log.error("[sync fallback] Failed to launch browser context: %s", exc)
            raise SerpCaptchaError(
                browser=serp_browser,
                message=f"Browser context launch failed: {exc}",
                wedged=True,
            ) from exc
        try:
            context.add_init_script(
                f"""
                Object.defineProperty(navigator, 'webdriver', {{get: () => undefined}});
                Object.defineProperty(navigator, 'hardwareConcurrency', {{get: () => {fingerprint['hardware_concurrency']} }});
                Object.defineProperty(navigator, 'deviceMemory', {{get: () => {fingerprint['device_memory']} }});
                """
            )
            page = context.new_page()
            start = 0
            while len(urls) < k:
                if device == "mobile":
                    search_url = f"{GOOGLE_URL}?q={quote_plus(keyword)}&client=ms-android-google&sourceid=chrome-mobile&start={start}"
                else:
                    search_url = f"{GOOGLE_URL}?q={quote_plus(keyword)}&hl=en&gl=us&start={start}"

                log.info(f"[sync fallback] Fetching: {search_url}")
                try:
                    page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
                except Exception as e:
                    log.error(f"[sync fallback] Error loading page: {e}. Retrying...")
                    time.sleep(3)
                    try:
                        page.goto(search_url, wait_until="networkidle", timeout=90000)
                    except Exception as e2:
                        log.error(f"[sync fallback] Failed after retry: {e2}.")
                        break

                if "/sorry/" in (page.url or ""):
                    _wait_for_manual_captcha_solve_sync(page, serp_browser)

                time.sleep(random.uniform(2, 4))
                raw_hrefs = page.evaluate(EXTRACT_JS)
                _check_captcha_sync(page, serp_browser, raw_hrefs)
                new_found = 0
                for h in raw_hrefs:
                    norm = normalize_url(h)
                    if is_organic_host(norm) and norm not in urls:
                        urls.append(norm)
                        new_found += 1

                log.info(f"[sync fallback] Found {new_found} new results (Total: {len(urls)})")
                if new_found == 0:
                    probe = page.evaluate(SERP_PAGE_PROBE_JS)
                    _log_serp_diagnostics(page.url, probe, 0)
                    if not probe.get("hasSearchRoot"):
                        raise SerpBlockedError(browser=serp_browser)
                if new_found == 0 or len(urls) >= k:
                    break
                start += CHUNK_SIZE
        finally:
            try:
                context.close()
            except Exception as close_err:
                log.warning("Ignoring sync context close error: %s", close_err)
    return urls[:k]

async def _scrape_google_results_async(
    keyword,
    k=20,
    headless=False,
    use_proxy=False,
    device="desktop",
    serp_browser=None,
    proxy_url=None,
):
    """Async version of scrape Google search results using Playwright."""
    serp_browser = serp_browser or default_serp_browser()
    try:
        async with async_playwright() as p:
            fingerprint = get_hardened_fingerprint(device, serp_browser)
            proxy_cfg = _build_proxy_cfg(use_proxy, proxy_url)
            user_data_dir = session_dir_for(serp_browser)

            urls = []
            context = None
            try:
                try:
                    context = await _launch_serp_context_async(
                        p, serp_browser, user_data_dir, headless, proxy_cfg, fingerprint
                    )
                except Exception as exc:
                    log.error("Failed to launch browser context: %s", exc)
                    raise SerpCaptchaError(
                        browser=serp_browser,
                        message=f"Browser context launch failed: {exc}",
                        wedged=True,
                    ) from exc

                await apply_stealth(context, fingerprint)
                page = await context.new_page()

                start = 0

                while len(urls) < k:
                    if device == "mobile":
                        search_url = f"{GOOGLE_URL}?q={quote_plus(keyword)}&client=ms-android-google&sourceid=chrome-mobile&start={start}"
                    else:
                        search_url = f"{GOOGLE_URL}?q={quote_plus(keyword)}&hl=en&gl=us&start={start}"

                    log.info(f"Fetching: {search_url}")

                    try:
                        await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
                    except Exception as e:
                        log.error(f"Error loading page: {e}. Retrying with longer timeout...")
                        await asyncio.sleep(3)
                        try:
                            await page.goto(search_url, wait_until="networkidle", timeout=90000)
                        except Exception as e2:
                            log.error(f"Failed to load page after retry: {e2}. Skipping...")
                            break

                    if "/sorry/" in (page.url or ""):
                        await _wait_for_manual_captcha_solve_async(page, serp_browser)

                    await asyncio.sleep(random.uniform(2, 4))

                    raw_hrefs = await page.evaluate(EXTRACT_JS)
                    await _check_captcha_async(page, serp_browser, raw_hrefs)
                    new_found = 0
                    for h in raw_hrefs:
                        norm = normalize_url(h)
                        if is_organic_host(norm) and norm not in urls:
                            urls.append(norm)
                            new_found += 1

                    log.info(f"Found {new_found} new results (Total: {len(urls)})")

                    if new_found == 0:
                        probe = await page.evaluate(SERP_PAGE_PROBE_JS)
                        _log_serp_diagnostics(page.url, probe, 0)
                        if not probe.get("hasSearchRoot"):
                            raise SerpBlockedError(browser=serp_browser)

                    if new_found == 0 or len(urls) >= k:
                        break

                    start += CHUNK_SIZE
            finally:
                if context is not None:
                    try:
                        await context.close()
                    except Exception as close_err:
                        log.warning("Ignoring browser context close error: %s", close_err)
        return urls[:k]
    except NotImplementedError:
        log.warning("Async Playwright unavailable on current event loop; using sync fallback.")
        return await asyncio.to_thread(
            _scrape_google_results_sync,
            keyword,
            k,
            headless,
            use_proxy,
            device,
            serp_browser,
            proxy_url,
        )

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")[:50]

def get_authority(domain):
    """
    Get authority score (0-10) based on DR from CSV.
    DR scale: 0-100 -> Authority scale: 0-10
    Default to 5 if domain not found.
    
    Handles subdomains by removing 'www.' prefix for CSV lookup.
    """
    # Remove 'www.' prefix if present to match CSV format
    lookup_domain = domain.replace('www.', '', 1) if domain.startswith('www.') else domain
    
    # First try lookup_domain, then try original domain
    if lookup_domain in DR_DATA:
        dr = DR_DATA[lookup_domain]
    elif domain in DR_DATA:
        dr = DR_DATA[domain]
    else:
        return 5  # Default to 5 if domain not found
    
    authority = int(round((dr / 100) * 10))
    # Clamp to 0-10 range
    authority = max(0, min(10, authority))
    return authority

def is_english(text):
    """Returns True if the text is English, False otherwise."""
    try:
        # We check a sample of the text to save time
        return detect(text[:1000]) == 'en'
    except:
        return False

# Replace your existing scrape_page_content and update main_pipeline tasks:

async def scrape_page_content(page, url):
    """
    Uses the existing Playwright page to scrape content,
    focusing only on title, description, headings, and paragraphs.
    """
    from serp_backends.page_backends import parse_page_html

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        html = await page.content()
        return parse_page_html(html, url)
    except Exception as e:
        log.error(f"Failed to scrape {url}: {e}")
        return {"url": url, "error": str(e)}

async def main_pipeline():
    # Handle non-interactive environments (Docker/Servers)
    try:
        keyword = input("Enter Keyword/Title: ")
        device = input("Search type (mobile/desktop): ").lower() or "desktop"
        proxy_choice = input("Use proxy? (y/n): ").lower() == 'y'
        headless_choice = input("Run headless? (y/n): ").lower() != 'n'  # Default to True
    except EOFError:
        log.warning("Non-interactive environment detected. Using defaults.")
        keyword = os.getenv("SEARCH_KEYWORD", "dog breeds")
        device = "desktop"
        proxy_choice = False
        headless_choice = True
    
    folder_name = clean_filename(keyword)
    os.makedirs(folder_name, exist_ok=True)
    
    print(f"Fetching top Google search results for: {keyword}")
    
    # --- PHASE 1: GOOGLE SERP (Your original logic untouched) ---
    results = await _scrape_google_results_async(keyword, k=20, headless=headless_choice, use_proxy=proxy_choice, device=device)
    
    # --- PHASE 2: SCRAPE INDIVIDUAL PAGES (Upgraded to Playwright) ---
    print(f"Scraping {len(results)} pages using browser context...")
    
    async with async_playwright() as p:
        fingerprint = get_hardened_fingerprint(device)
        # Note: proxy is NOT used for scraping individual pages, only for SERP
        
        browser = await p.chromium.launch(headless=headless_choice)
        context = await browser.new_context(
            user_agent=fingerprint['user_agent'],
            viewport=fingerprint['viewport']
        )
        await apply_stealth(context, fingerprint)
        page = await context.new_page()

        for url in results:
            # We process sequentially to avoid overwhelming the system/proxy
            # or you can open multiple pages in tabs if your hardware allows.
            data = await scrape_page_content(page, url)
            
            domain = urlparse(data['url']).netloc
            file_id = clean_filename(domain + "_" + str(hash(data['url']))[:5])
            
            with open(f"{folder_name}/{file_id}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            
            status = "Success" if "content" in data else "Error"
            print(f"Processed: {domain} -> {status}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main_pipeline())
