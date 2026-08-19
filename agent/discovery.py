"""
agent/discovery.py — LLM-driven discovery engine.

The agent loop:
  1. Take a screenshot + extract accessibility tree from the current page
  2. Send to the LLM (GPT-4o vision or Claude with vision) with available tools
  3. LLM calls one of: navigate, fill, click, select, wait, extract, mark_done, escalate
  4. We execute the action, record it as a Step, evaluate any checkpoint
  5. Repeat until LLM calls mark_done or max_steps exceeded

Tool definitions mirror the artifact Step types so recording is direct.
The LLM never runs during replay — this is the ONLY place the LLM is used.

Supports: OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, Google Gemini 1.5 Pro, OpenRouter
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Optional

from agent.artifact import Artifact, ArtifactSafety, Step, Locator, Checkpoint
from agent.artifact import ArtifactStore
from agent.browser import BrowserManager, evaluate_checkpoint
from agent.safety import Guardrail, Policy, redact_step_value
from agent.escalation import EscalationManager
from agent.logger import RunLogger


# ─── LLM Client Factory ───────────────────────────────────────────────────────

def get_llm_client():
    """Return (client, model, provider) based on environment."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "openrouter":
        from openai import AsyncOpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY in your .env file.")
        # Default to a vision-capable model available on OpenRouter
        # google/gemini-flash-1.5 has high free limits and supports vision
        model = os.environ.get("LLM_MODEL", "google/gemini-flash-1.5")
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/interface-ai-assignment",
                "X-Title": "Computer-Use Automation Agent",
            },
        )
        # OpenRouter uses the OpenAI call path
        return client, model, "openrouter"

    elif provider == "openai":
        from openai import AsyncOpenAI
        model = os.environ.get("LLM_MODEL", "gpt-4o")
        return AsyncOpenAI(), model, "openai"

    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        model = os.environ.get("LLM_MODEL", "claude-3-5-sonnet-20241022")
        return AsyncAnthropic(), model, "anthropic"

    elif provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = os.environ.get("LLM_MODEL", "gemini-1.5-pro")
        return genai, model, "google"

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use openai, anthropic, or google.")


# ─── Tool Definitions (OpenAI format — converted for Anthropic/Google) ────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate the browser to a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to navigate to."},
                    "description": {"type": "string", "description": "Why you are navigating here."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill",
            "description": "Type text into an input field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "locator_strategy": {"type": "string", "enum": ["aria-label", "data-testid", "role", "text", "css"]},
                    "locator_value": {"type": "string"},
                    "value": {"type": "string", "description": "Text to type. Use {{param_name}} for parameterized values."},
                    "description": {"type": "string"},
                    "is_param": {"type": "boolean", "description": "True if this value should be a parameter (varies per run)."},
                    "param_name": {"type": "string", "description": "Parameter name if is_param is true."},
                },
                "required": ["locator_strategy", "locator_value", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click a button, link, or element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "locator_strategy": {"type": "string", "enum": ["aria-label", "data-testid", "role", "text", "css"]},
                    "locator_value": {"type": "string"},
                    "locator_name": {"type": "string", "description": "For role strategy: the accessible name."},
                    "description": {"type": "string"},
                    "checkpoint_type": {
                        "type": "string",
                        "enum": ["element_visible", "url_contains", "text_contains", "none"],
                        "description": "What to assert after clicking.",
                    },
                    "checkpoint_value": {"type": "string", "description": "The value for the checkpoint."},
                },
                "required": ["locator_strategy", "locator_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select",
            "description": "Select an option from a <select> dropdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "locator_strategy": {"type": "string", "enum": ["aria-label", "data-testid", "role", "text", "css"]},
                    "locator_value": {"type": "string"},
                    "value": {"type": "string", "description": "Option value to select."},
                    "description": {"type": "string"},
                },
                "required": ["locator_strategy", "locator_value", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract",
            "description": "Extract text from an element and record it as an output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "locator_strategy": {"type": "string", "enum": ["aria-label", "data-testid", "role", "text", "css"]},
                    "locator_value": {"type": "string"},
                    "output_key": {"type": "string", "description": "Key name for this output (e.g. 'account_balance')."},
                    "description": {"type": "string"},
                },
                "required": ["locator_strategy", "locator_value", "output_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a number of milliseconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ms": {"type": "integer", "description": "Milliseconds to wait."},
                },
                "required": ["ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_done",
            "description": "Signal that the goal has been completed successfully.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Brief summary of what was accomplished."},
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of output keys that were extracted.",
                    },
                    "params": {
                        "type": "object",
                        "description": "Map of param_name → type hint (e.g. 'string').",
                    },
                },
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": "Request human intervention when you are stuck and cannot proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why you cannot proceed."},
                },
                "required": ["reason"],
            },
        },
    },
]


# ─── System Prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(goal: str, base_url: str) -> str:
    return f"""You are a computer-use automation agent. Your job is to operate a web application to achieve a goal, then record the steps you took as a replayable artifact.

GOAL: {goal}
BASE URL: {base_url}

INSTRUCTIONS:
1. Look at the current page (screenshot + accessibility tree) and decide the next action.
2. Call exactly ONE tool per turn.
3. PREFER navigating directly to a page URL rather than clicking nav links — it is more reliable.
   For example, to go to members use: navigate to {base_url}/members
4. Prefer stable locator strategies in this order: data-testid > aria-label > role > text > css.
5. If a locator fails, you will see the error in the next turn. Try a different strategy or navigate by URL.
6. When you fill in a value that should vary per run (like a member name), set is_param=true and param_name (e.g. 'member_name'). Use {{{{param_name}}}} as the value.
7. When you have accomplished the goal and extracted all needed outputs, call mark_done.
8. If you are stuck after 3 failed attempts, call escalate.
9. NEVER navigate outside the allowed domain: {base_url}

KEY PAGES:
- Member search: {base_url}/members
- Fund transfer: {base_url}/transfer
- Loan lookup: {base_url}/loans

IMPORTANT: You are recording steps to be replayed deterministically. Choose actions that will work reliably on future runs. Avoid nth-child or absolute XPaths."""


# ─── Discovery Engine ─────────────────────────────────────────────────────────

MAX_STEPS = 30  # Hard limit on LLM turns to prevent runaway costs


def _make_fallbacks(primary_strategy: str, primary_value: str) -> list[dict]:
    """Generate fallback locator strategies for a primary locator.

    Discovery records these so replay has multiple ways to find an element
    if the primary locator fails (e.g., after a UI refactor).
    """
    fallbacks = []
    # Always try text as a fallback (human-readable, survives CSS changes)
    if primary_strategy != "text" and primary_value and len(primary_value) < 60:
        fallbacks.append({"strategy": "text", "value": primary_value})
    # Add CSS data-testid fallback if value looks like a testid
    if primary_strategy not in ("data-testid", "css"):
        slug = primary_value.lower().replace(" ", "-").replace("_", "-")
        if slug:
            fallbacks.append({"strategy": "css", "value": f"[data-testid*='{slug[:30]}']"})
    # Add aria-label fallback
    if primary_strategy not in ("aria-label",) and primary_value:
        fallbacks.append({"strategy": "aria-label", "value": primary_value})
    return fallbacks[:2]  # limit to 2 fallbacks to keep artifacts concise




class DiscoveryEngine:
    """
    LLM-driven agent that discovers a UI flow and produces an Artifact.

    Usage:
        engine = DiscoveryEngine(goal="Find Jane Doe's balance", base_url="http://localhost:5000")
        artifact = await engine.run()
    """

    def __init__(
        self,
        goal: str,
        base_url: str,
        evidence_dir: str = "evidence",
        artifact_dir: str = "artifacts",
        headless: bool = False,
        allow_irreversible: bool = False,
    ):
        self.goal = goal
        self.base_url = base_url.rstrip("/")
        self.evidence_dir = evidence_dir
        self.artifact_dir = artifact_dir
        self.headless = headless
        self.allow_irreversible = allow_irreversible

        self.run_id = str(uuid.uuid4())
        self.logger = RunLogger(self.run_id, "discovery", evidence_dir)
        self.escalator = EscalationManager(self.run_id, goal)

        # Safety enforcement — same as replay, not just prompt instructions
        policy = Policy.from_env()
        if allow_irreversible:
            policy.irreversible_policy = "allow"
        else:
            policy.irreversible_policy = "block"
        self.guardrail = Guardrail(policy)

        self._steps: list[Step] = []
        self._param_defs: list = []   # list[ArtifactParam]
        self._output_defs: list = []  # list[ArtifactOutput]
        self._step_seq = 0
        self._messages: list[dict] = []

        self._client, self._model, self._provider = get_llm_client()
        self._browser: Optional[BrowserManager] = None

    async def run(self) -> Artifact:
        self.logger.info(f"Starting discovery: {self.goal}")
        self.logger.info(f"LLM: {self._provider}/{self._model}")

        self._browser = BrowserManager(headless=self.headless, evidence_dir=self.evidence_dir)
        await self._browser.start()

        try:
            artifact = await self._agent_loop()
        finally:
            await self._browser.stop()

        # Save artifact (validate() called inside store.save())
        store = ArtifactStore(self.artifact_dir)
        path = store.save(artifact)
        self.logger.info(f"Artifact saved: {path}")
        self.logger.run_end("success", {"artifact_id": artifact.id, "steps": len(artifact.steps)})
        return artifact

    async def _agent_loop(self) -> Artifact:
        """Main LLM → action → observe loop."""
        # Navigate to base URL and record as step 1
        await self._browser.navigate(self.base_url)
        self.logger.info(f"Navigated to {self.base_url}")
        self._step_seq = 1
        self._steps.append(Step(
            seq=1,
            action="navigate",
            url=self.base_url,
            description="Initial navigation to base URL",
        ))

        last_error: str = ""
        completed = False

        for turn in range(MAX_STEPS):
            # Perceive
            page_text = await self._browser.get_page_text()
            screenshot_b64 = await self._browser.get_screenshot_b64()
            current_url = self._browser.page.url

            # Build message with current state (include last error if any)
            user_content = self._build_perception_message(page_text, current_url, turn, last_error)
            last_error = ""  # reset

            # Call LLM
            tool_name, tool_args, prompt_tokens, completion_tokens = await self._call_llm(
                user_content, screenshot_b64
            )

            self.logger.llm_call(
                self._model, prompt_tokens, completion_tokens, [tool_name]
            )

            # Handle terminal tools
            if tool_name == "mark_done":
                self.logger.info(f"mark_done: {tool_args.get('summary', '')}")
                completed = True
                # Build typed output and param definitions
                from agent.artifact import ArtifactParam, ArtifactOutput
                for out_key in tool_args.get("outputs", []):
                    self._output_defs.append(ArtifactOutput(key=out_key, type="string"))
                for p_name, p_type in tool_args.get("params", {}).items():
                    # Only add if not already declared
                    existing = [p.name for p in self._param_defs]
                    if p_name not in existing:
                        self._param_defs.append(
                            ArtifactParam(name=p_name, type="string")
                        )
                break

            if tool_name == "escalate":
                # During discovery: take screenshot, log, feed error back to LLM
                reason = tool_args.get("reason", "Agent requested escalation")
                screenshot = await self._browser.screenshot(
                    f"escalation_discovery_{self.run_id[:8]}"
                )
                self.logger.info(f"LLM escalated (discovery — non-blocking): {reason}")
                last_error = (
                    f"You called escalate with: '{reason}'. "
                    f"In discovery mode, escalation is not available. "
                    f"Look at the current page carefully and try a different approach — "
                    f"use 'extract' to pull the balance value from data-testid='total-balance', "
                    f"or navigate directly to a specific URL like {self.base_url}/members/M001."
                )
                continue

            # Execute action and record step (errors feed back to LLM, not crash)
            step, err = await self._execute_and_record(tool_name, tool_args, current_url)
            if err:
                last_error = err
            elif step:
                self._steps.append(step)

        if not completed:
            raise RuntimeError(
                f"Discovery reached MAX_STEPS ({MAX_STEPS}) without calling mark_done. "
                f"No artifact saved — discovery failed."
            )

        # Build artifact with typed declarations
        from agent.artifact import ArtifactSafety
        artifact = Artifact(
            goal=self.goal,
            surface="web",
            base_url=self.base_url,
            steps=self._steps,
            param_defs=self._param_defs,
            output_defs=self._output_defs,
            discovery_model=f"{self._provider}/{self._model}",
            safety=ArtifactSafety(
                allowed_domains=[self.base_url.split("://")[-1].split("/")[0]],
                reversibility="read-only" if not any(
                    s.action in {"click", "fill", "select"} for s in self._steps
                ) else "write-reversible",
            ),
        )
        return artifact

    async def _call_llm(
        self, user_content: str, screenshot_b64: str
    ) -> tuple[str, dict, int, int]:
        """Call the LLM and return (tool_name, tool_args, prompt_tokens, completion_tokens)."""
        system = build_system_prompt(self.goal, self.base_url)

        if self._provider in ("openai", "openrouter"):
            return await self._call_openai(system, user_content, screenshot_b64)
        elif self._provider == "anthropic":
            return await self._call_anthropic(system, user_content, screenshot_b64)
        else:
            raise ValueError(f"Provider {self._provider} not yet fully implemented")

    async def _call_openai(self, system: str, user_content: str, screenshot_b64: str) -> tuple:
        # Detect image format — PIL saves as JPEG, raw Playwright as PNG
        try:
            import base64 as _b64
            raw = _b64.b64decode(screenshot_b64[:16])  # check magic bytes
            img_mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
        except Exception:
            img_mime = "image/png"

        self._messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_mime};base64,{screenshot_b64}"},
                },
                {"type": "text", "text": user_content},
            ],
        })

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}] + self._messages,
            tools=TOOLS,
            tool_choice="required",
            max_tokens=1024,
        )

        msg = response.choices[0].message
        self._messages.append(msg.model_dump())

        tool_call = msg.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # Add tool result to messages
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Executed: {tool_name}",
        })

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        return tool_name, tool_args, prompt_tokens, completion_tokens

    async def _call_anthropic(self, system: str, user_content: str, screenshot_b64: str) -> tuple:
        """Call Anthropic Claude with vision + tool use."""
        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for t in TOOLS:
            fn = t["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": fn["parameters"],
            })

        # Build messages
        self._messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot_b64,
                    },
                },
                {"type": "text", "text": user_content},
            ],
        })

        response = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=self._messages,
            tools=anthropic_tools,
            max_tokens=1024,
        )

        # Extract tool use
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if not tool_use:
            raise RuntimeError("LLM did not call any tool. Response: " + str(response.content))

        tool_name = tool_use.name
        tool_args = tool_use.input

        # Add assistant + tool result to messages
        self._messages.append({"role": "assistant", "content": response.content})
        self._messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": f"Executed: {tool_name}"}],
        })

        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        return tool_name, tool_args, prompt_tokens, completion_tokens

    async def _execute_and_record(
        self, tool_name: str, args: dict, current_url: str
    ) -> tuple[Optional[Step], Optional[str]]:
        """Execute a tool call, enforce safety, and record it as a Step.

        Returns (step, None) on success, (None, error_str) on failure.
        Step seq is only incremented on success to avoid gaps.
        """
        description = args.get("description", "")

        # Screenshot before action for evidence
        next_seq = self._step_seq + 1
        await self._browser.screenshot(f"discovery_step{next_seq:02d}_before")

        step = None
        try:
            if tool_name == "navigate":
                url = args["url"]
                # Safety check
                violation = self.guardrail.check_navigate(url)
                if violation:
                    return None, f"Safety violation: {violation.detail}"
                self._step_seq += 1
                seq = self._step_seq
                self.logger.step_start(seq, tool_name, description)
                await self._browser.navigate(url)
                step = Step(seq=seq, action="navigate", url=url, description=description)

            elif tool_name == "fill":
                primary_strategy = args["locator_strategy"]
                primary_value = args["locator_value"]
                # Auto-generate fallbacks for robustness
                fallbacks = _make_fallbacks(primary_strategy, primary_value)
                loc = Locator(
                    strategy=primary_strategy,
                    value=primary_value,
                    fallbacks=fallbacks,
                )
                value = args["value"]
                param_name = args.get("param_name", "")
                is_param = args.get("is_param", False)
                if is_param and param_name:
                    from agent.artifact import ArtifactParam
                    existing = [p.name for p in self._param_defs]
                    if param_name not in existing:
                        self._param_defs.append(ArtifactParam(name=param_name, type="string"))
                    # Record parameterized placeholder in artifact
                    artifact_value = f"{{{{{param_name}}}}}"
                else:
                    artifact_value = redact_step_value(value)
                # Safety check
                violation, _ = self.guardrail.check_action("fill", current_url, value)
                if violation:
                    return None, f"Safety violation: {violation.detail}"
                self._step_seq += 1
                seq = self._step_seq
                self.logger.step_start(seq, tool_name, description)
                await self._browser.fill(loc, value)  # use original for actual fill
                step = Step(
                    seq=seq, action="fill", locator=loc, value=artifact_value,
                    description=description + (f" [param:{param_name}]" if is_param else ""),
                )

            elif tool_name == "click":
                primary_strategy = args["locator_strategy"]
                primary_value = args["locator_value"]
                fallbacks = _make_fallbacks(primary_strategy, primary_value)
                loc = Locator(
                    strategy=primary_strategy,
                    value=primary_value,
                    name=args.get("locator_name"),
                    fallbacks=fallbacks,
                )
                # Safety check
                violation, reversibility = self.guardrail.check_action("click", current_url, primary_value)
                if violation and (reversibility == "write-irreversible" and not self.allow_irreversible):
                    return None, f"Safety violation: {violation.detail}"
                self._step_seq += 1
                seq = self._step_seq
                self.logger.step_start(seq, tool_name, description)
                await self._browser.click(loc)
                checkpoint = None
                cp_type = args.get("checkpoint_type", "none")
                cp_value = args.get("checkpoint_value", "")
                if cp_type and cp_type != "none" and cp_value:
                    from agent.artifact import Checkpoint
                    if cp_type == "url_contains":
                        checkpoint = Checkpoint(type="url_contains", value=cp_value)
                    elif cp_type == "element_visible":
                        checkpoint = Checkpoint(
                            type="element_visible",
                            locator=Locator(strategy="css", value=cp_value),
                        )
                    elif cp_type == "text_contains":
                        checkpoint = Checkpoint(type="text_contains", value=cp_value)
                    if checkpoint:
                        ok, detail = await evaluate_checkpoint(self._browser.page, checkpoint)
                        if ok:
                            self.logger.checkpoint_ok(seq, checkpoint.type)
                        else:
                            # Log as failure but continue — LLM may recover
                            self.logger.checkpoint_fail(seq, checkpoint.type, detail)
                            checkpoint = None  # don't persist a known-failing checkpoint
                step = Step(
                    seq=seq, action="click", locator=loc, checkpoint=checkpoint,
                    description=description
                )

            elif tool_name == "select":
                loc = Locator(strategy=args["locator_strategy"], value=args["locator_value"])
                self._step_seq += 1
                seq = self._step_seq
                self.logger.step_start(seq, tool_name, description)
                await self._browser.select_option(loc, args["value"])
                step = Step(
                    seq=seq, action="select", locator=loc, value=args["value"],
                    description=description
                )

            elif tool_name == "extract":
                primary_strategy = args["locator_strategy"]
                primary_value = args["locator_value"]
                fallbacks = _make_fallbacks(primary_strategy, primary_value)
                loc = Locator(
                    strategy=primary_strategy, value=primary_value, fallbacks=fallbacks
                )
                self._step_seq += 1
                seq = self._step_seq
                self.logger.step_start(seq, tool_name, description)
                text = await self._browser.extract_text(loc)
                output_key = args["output_key"]
                self.logger.info(f"Extracted '{output_key}': {text[:80]}")
                from agent.artifact import ArtifactOutput
                existing_keys = [o.key for o in self._output_defs]
                if output_key not in existing_keys:
                    self._output_defs.append(ArtifactOutput(key=output_key, type="string"))
                step = Step(
                    seq=seq, action="extract", locator=loc, output_key=output_key,
                    description=description
                )

            elif tool_name == "wait":
                ms = args.get("ms", 1000)
                self._step_seq += 1
                seq = self._step_seq
                self.logger.step_start(seq, tool_name, description)
                await self._browser.wait(ms)
                step = Step(seq=seq, action="wait", wait_ms=ms, description=description)

            else:
                return None, f"Unknown tool: {tool_name}"

            self.logger.step_ok(self._step_seq, tool_name)

        except Exception as e:
            screenshot = await self._browser.screenshot(
                f"discovery_error_step{next_seq:02d}"
            )
            err_str = str(e)
            self.logger.step_fail(next_seq, tool_name, err_str, screenshot)
            # Return error to caller — LLM will get the error as context and retry
            return None, err_str

        return step, None

    def _build_perception_message(
        self, page_text: str, current_url: str, turn: int, last_error: str = ""
    ) -> str:
        error_context = ""
        if last_error:
            error_context = (
                f"\n⚠️ PREVIOUS ACTION FAILED: {last_error}\n"
                f"Try a different locator strategy (text, role, css, data-testid) "
                f"or navigate directly to the target page by URL.\n"
            )
        return (
            f"Turn {turn + 1}/{MAX_STEPS}\n"
            f"Current URL: {current_url}\n"
            f"{error_context}\n"
            f"Accessibility tree:\n{page_text[:3000]}\n\n"
            f"What is your next action to achieve the goal: {self.goal}?"
        )
