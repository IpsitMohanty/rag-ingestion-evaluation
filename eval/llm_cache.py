"""Disk cache for phase 4's LLM calls, keyed on (node, model, prompt,
run_index) so re-running the harness during development (bug fixes,
metric changes) doesn't re-spend on a call already made *for that run*.
One JSON file, loaded/saved whole -- sized for a few hundred entries
(eval/METHODOLOGY.md #16's ~459-call envelope), not a general-purpose
cache.

run_index is load-bearing, not decorative: eval/METHODOLOGY.md #15 needs
3 independent samples per arm to measure run-to-run variance. Retrieval
is deterministic and queries are fixed, so without run_index in the key,
every run after the first would be a guaranteed cache hit on run 1's
response -- the 3-run range would be exactly zero on every metric by
construction, not because the API is actually stable, and every
judge-vs-threshold gap would then trivially "exceed" a variance bar of
zero. run_index makes each of the 3 runs a genuinely fresh sampling
event while still caching (and not re-spending on) repeated invocations
of that same run_index across separate harness executions.

Caching is deliberately kept out of src/adapters/agentic.py: the graph
only knows about route_llm/judge_llm's .invoke() contract, not about
caching or which run it's on. CachedStructuredLLM below wraps a real
structured-output runnable so the graph never needs to know the
difference between a fresh call and a cache hit.
"""
import hashlib
import json
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent / ".llm_cache.json"


def _key(node: str, model: str, prompt: str, run_index: int) -> str:
    return hashlib.sha256(f"{node}||{model}||run{run_index}||{prompt}".encode("utf-8")).hexdigest()


def load_cache(path: Path = CACHE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, path: Path = CACHE_PATH) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


class CachedStructuredLLM:
    """Wraps a structured-output runnable (e.g.
    ChatOpenAI(...).with_structured_output(RouteDecision)) with the disk
    cache above. `schema` is the Pydantic model class, needed to
    reconstruct a cached hit from its serialized dict. Tracks call_count
    (only real, non-cached invocations) so the harness can compare actual
    spend against the approved ~459-call / ~$0.06 estimate as it runs.

    `run_index` is a plain, settable attribute (not a constructor-only
    value) so eval/agentic_sweep.py's run_arm_c can update it once per
    run, reusing the same wrapper instance across all 3 runs rather than
    rebuilding the whole graph 3 times.
    """

    def __init__(self, runnable, schema, node: str, model_name: str, cache: dict, run_index: int = 0):
        self._runnable = runnable
        self._schema = schema
        self._node = node
        self._model_name = model_name
        self._cache = cache
        self.run_index = run_index
        self.call_count = 0

    def invoke(self, prompt: str):
        key = _key(self._node, self._model_name, prompt, self.run_index)
        cached = self._cache.get(key)
        if cached is not None:
            return self._schema(**cached)

        result = self._runnable.invoke(prompt)
        self.call_count += 1
        self._cache[key] = result.model_dump()
        return result
