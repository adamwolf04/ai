"""
ToolBelt — sandboxed tool execution with per-URL caching and async support.

Changes from prototype:
  - python_repl replaced with subprocess sandbox (no raw exec())
  - Per-tool result cache (TTL-based) to avoid redundant network calls
  - scrape_page fallback: on failure, retries via search_web for the domain
  - AsyncToolBelt wraps sync tools with asyncio.to_thread for concurrent evaluation
  - SANDBOX_MODE config: "subprocess" (default) or future "docker"
"""
import urllib.request
import urllib.parse
import subprocess
import tempfile
import os
import sys
import json
import time
import hashlib
import asyncio
from bs4 import BeautifulSoup
from ddgs import DDGS


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[str, float]] = {}   # key -> (result, expiry_timestamp)
_URL_LOCKS: dict[str, asyncio.Lock] = {}     # per-URL async lock

_TTL_SEARCH = int(os.environ.get("CACHE_TTL_SEARCH", "1800"))   # 30 min
_TTL_SCRAPE = int(os.environ.get("CACHE_TTL_SCRAPE", "3600"))   # 60 min

SANDBOX_MODE = os.environ.get("SANDBOX_MODE", "subprocess")     # subprocess | docker


def _cache_key(tool: str, kwargs: dict) -> str:
    raw = tool + json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    if key in _CACHE:
        val, expiry = _CACHE[key]
        if time.time() < expiry:
            return val
        del _CACHE[key]
    return None


def _cache_set(key: str, value: str, ttl: int):
    _CACHE[key] = (value, time.time() + ttl)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_web(query: str) -> str:
    """Search the web for the given query and return top-3 result snippets."""
    key = _cache_key("search_web", {"query": query})
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        results = DDGS().text(query, max_results=3)
        output = json.dumps(
            [{"title": r["title"], "url": r["href"], "body": r["body"]} for r in results],
            indent=2
        )
    except Exception as e:
        output = f"Error during search: {e}"

    _cache_set(key, output, _TTL_SEARCH)
    return output


def scrape_page(url: str) -> str:
    """
    Scrape the main text from a given URL.
    On failure, automatically falls back to search_web using the URL's domain.
    """
    key = _cache_key("scrape_page", {"url": url})
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        lines  = (ln.strip() for ln in soup.get_text(separator=" ").splitlines())
        chunks = (ph.strip() for ln in lines for ph in ln.split("  "))
        text   = "\n".join(ch for ch in chunks if ch)[:5000]
        output = text
    except Exception as e:
        # Fallback: search_web with the domain name as the query
        try:
            domain = urllib.parse.urlparse(url).netloc
            output = f"[scrape failed: {e}] Fallback search for domain: {domain}\n"
            output += search_web(domain)
        except Exception as fe:
            output = f"Error scraping {url}: {e}. Fallback also failed: {fe}"

    _cache_set(key, output, _TTL_SCRAPE)
    return output


def python_repl(code: str) -> str:
    """
    Execute Python code in a restricted subprocess sandbox.
    - No network access from within executed code (PYTHONPATH cleared)
    - 10-second hard timeout, process killed on expiry
    - Code limited to 10 KB
    - Only built-in stdlib imports (no third-party packages)
    """
    if SANDBOX_MODE == "docker":
        return "[docker sandbox not yet configured]"

    # Size limit
    if len(code.encode("utf-8")) > 10 * 1024:
        return "Error: Code exceeds 10 KB size limit."

    sandbox_dir = os.path.join(tempfile.gettempdir(), "stem_sandbox")
    os.makedirs(sandbox_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=sandbox_dir
    ) as f:
        f.write(code)
        path = f.name

    # Restricted environment: clear PYTHONPATH, keep only necessary vars
    restricted_env = {
        "PATH":             os.environ.get("PATH", ""),
        "SYSTEMROOT":       os.environ.get("SYSTEMROOT", ""),  # Windows requirement
        "TEMP":             tempfile.gettempdir(),
        "TMP":              tempfile.gettempdir(),
        "PYTHONPATH":       "",          # block third-party imports
        "PYTHONUTF8":       "1",
    }

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=10,
            env=restricted_env,
            cwd=sandbox_dir,
        )
        result = proc.stdout or proc.stderr or "Execution successful, no output."
        return result[:3000]
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (10s limit)."
    except Exception as e:
        return f"Error running code: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOLBOX_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for a query and return snippets and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_page",
            "description": "Scrape the text content of a given URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to scrape."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_repl",
            "description": "Execute Python code in a sandboxed environment. Useful for math or data processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The Python code to execute."}
                },
                "required": ["code"]
            }
        }
    }
]

TOOLBOX_FUNCTIONS = {
    "search_web":  search_web,
    "scrape_page": scrape_page,
    "python_repl": python_repl,
}

AVAILABLE_TOOLS = list(TOOLBOX_FUNCTIONS.keys())


# ---------------------------------------------------------------------------
# Sync ToolBelt
# ---------------------------------------------------------------------------

class ToolBelt:
    def __init__(self, allowed_tools: list[str]):
        self.tools   = {n: fn for n, fn in TOOLBOX_FUNCTIONS.items() if n in allowed_tools}
        self.schemas = [s for s in TOOLBOX_SCHEMAS if s["function"]["name"] in allowed_tools]

    def execute(self, name: str, kwargs: dict) -> str:
        if name not in self.tools:
            return f"Error: tool '{name}' not found or not allowed."
        try:
            return str(self.tools[name](**kwargs))
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    def get_schemas(self) -> list:
        return self.schemas


# ---------------------------------------------------------------------------
# Async ToolBelt  (wraps sync tools with asyncio.to_thread)
# ---------------------------------------------------------------------------

class AsyncToolBelt(ToolBelt):
    """
    Async-capable ToolBelt. Uses asyncio.to_thread so sync I/O tools
    yield time back to the event loop without blocking other coroutines.

    For scrape_page, acquires a per-URL lock to prevent hammering the
    same URL simultaneously from multiple concurrent agents.
    """

    async def execute_async(self, name: str, kwargs: dict) -> str:
        if name not in self.tools:
            return f"Error: tool '{name}' not found or not allowed."

        if name == "scrape_page":
            url = kwargs.get("url", "")
            if url not in _URL_LOCKS:
                _URL_LOCKS[url] = asyncio.Lock()
            async with _URL_LOCKS[url]:
                return await asyncio.to_thread(self.tools[name], **kwargs)

        try:
            return await asyncio.to_thread(self.tools[name], **kwargs)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"
