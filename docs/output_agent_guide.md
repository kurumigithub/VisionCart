# VisionCart — Output Agent Guide
`src/agents/output.py`

---

## What this agent does

Your agent is the last step in the pipeline. It receives a structured payload from the ranker/critic agents — ranked products with scores, style tags, and match reasoning — and synthesizes that into a clean, human-readable response the user can act on.

**Inputs it receives** (from `graph/state.py`):
- `state["style_profile"]` — the aesthetic embedding produced by stylist.py (described in text, e.g. "warm earth tones, minimalist silhouettes, textural contrast")
- `state["ranked_products"]` — a list of product dicts, ordered by similarity score
- `state["critic_notes"]` — optional: any products the critic flagged or filtered out

**Output it produces**:
- A formatted string (or structured dict, depending on how your team decides to surface the UI) that narrates the results to the user

---

## Core responsibility

Translate machine outputs into natural language. Specifically:

1. **Summarize the style profile** in 1–2 sentences so the user can confirm the system understood their aesthetic
2. **Narrate the top results** — not just list them, but explain *why* each match fits
3. **Surface the ranking logic** in plain English (e.g. "This jacket scored highest for silhouette and color palette alignment")
4. **Flag edge cases** if the critic rejected something useful, or if results are sparse

---

## How agents work in LangGraph (quick primer)

VisionCart uses LangGraph to coordinate agents. Each agent is a node in a graph. They communicate through a shared `state` object — a Python dict that every agent reads from and writes to.

Your agent receives the state, reads from it, calls an LLM to generate a narrative, and writes the output string back into the state.

```
state flows in → output.py reads it → calls Claude API → writes result to state → state flows out
```

You don't need to worry about routing — that's handled in `graph/state.py`. You just need to write the node function.

---

## The code pattern you'll follow

Every LangGraph agent node is a Python function with this signature:

```python
def run(state: dict) -> dict:
    # read from state
    # do your logic
    # return a dict with your new keys
    return {"output_text": "..."}
```

The returned dict gets merged into the shared state automatically by LangGraph.

---

## Starter code

```python
# src/agents/output.py

from anthropic import Anthropic

client = Anthropic()

SYSTEM_PROMPT = """
You are a personal shopping assistant for a visual-first retail app.
Your job is to translate AI-ranked product results into a warm,
concise, human-readable summary that explains why the results
match the user's aesthetic and helps them make a decision.

Format your response as:
1. A 1-2 sentence interpretation of the user's style profile
2. Top 3-5 products with a brief explanation of why each fits
3. An optional note if results were sparse or any products were filtered

Keep the tone conversational and specific — reference actual
aesthetic details like color, texture, silhouette, or vibe.
Do not use filler phrases like "Great news!" or "I found some products."
"""

def format_products_for_prompt(ranked_products: list) -> str:
    """Convert ranked product list into a readable block for the prompt."""
    lines = []
    for i, p in enumerate(ranked_products[:5], 1):
        lines.append(
            f"{i}. {p.get('name', 'Unknown Product')}\n"
            f"   Score: {p.get('score', 'N/A')}\n"
            f"   Tags: {', '.join(p.get('tags', []))}\n"
            f"   URL: {p.get('url', '')}\n"
            f"   Price: {p.get('price', 'N/A')}"
        )
    return "\n\n".join(lines)


def run(state: dict) -> dict:
    """
    LangGraph node: generate human-readable output from ranked results.

    Reads from state:
        - style_profile (str): text description of the user's aesthetic
        - ranked_products (list): products sorted by similarity score
        - critic_notes (str, optional): any rejections or flags

    Writes to state:
        - output_text (str): the final user-facing response
    """
    style_profile = state.get("style_profile", "No style profile provided.")
    ranked_products = state.get("ranked_products", [])
    critic_notes = state.get("critic_notes", "")

    if not ranked_products:
        return {"output_text": "We couldn't find products that matched your style profile. Try adding more images to your board."}

    products_block = format_products_for_prompt(ranked_products)

    user_message = f"""
Style profile: {style_profile}

Ranked products:
{products_block}

{f"Critic notes: {critic_notes}" if critic_notes else ""}

Please generate the user-facing summary.
"""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    output_text = response.content[0].text

    return {"output_text": output_text}
```

---

## Running this locally with Claude Code

Claude Code is a CLI tool that lets you work with Claude directly in your terminal alongside your codebase. It's the most practical way to build this if you're not deeply technical.

### Setup steps

```bash
# 1. Install Claude Code
npm install -g @anthropic/claude-code

# 2. Navigate to your project
cd VisionCart

# 3. Set your API key (get one from console.anthropic.com)
export ANTHROPIC_API_KEY=your_key_here

# 4. Start Claude Code
claude
```

Once inside Claude Code, you can say things like:
- `"Look at src/agents/output.py and help me test this with a sample state dict"`
- `"Add error handling if ranked_products is empty or malformed"`
- `"Write a unit test for the run() function using mock state data"`
- `"Help me understand how state.py passes data into my agent"`

Claude Code can read your whole repo, so it'll give you context-aware help rather than generic answers.

---

## Testing your agent in isolation

Before wiring into the full graph, test your node directly:

```python
# tests/test_output_agent.py

import sys
sys.path.insert(0, "src")

from agents.output import run

mock_state = {
    "style_profile": "Warm earth tones, relaxed Japanese streetwear silhouettes, natural textures like linen and washed denim, minimal branding.",
    "ranked_products": [
        {
            "name": "Uniqlo Wide Linen Trousers",
            "score": 0.91,
            "tags": ["linen", "wide-leg", "neutral", "minimalist"],
            "url": "https://uniqlo.com/...",
            "price": "$49.90"
        },
        {
            "name": "Mango Washed Denim Overshirt",
            "score": 0.84,
            "tags": ["denim", "oversized", "earth tone", "texture"],
            "url": "https://mango.com/...",
            "price": "$69.99"
        }
    ],
    "critic_notes": "Filtered 2 products that matched color but had prominent logo branding."
}

result = run(mock_state)
print(result["output_text"])
```

Run it with:
```bash
cd VisionCart
python tests/test_output_agent.py
```

---

## Connecting to the LangGraph graph

Once your agent is working in isolation, your team's graph maintainer (whoever owns `graph/state.py`) will wire it in as the final node:

```python
# graph/state.py (rough example — your team's graph code may differ)

from langgraph.graph import StateGraph
from agents import stylist, procurement, ranker, critic, output

graph = StateGraph(dict)
graph.add_node("stylist", stylist.run)
graph.add_node("procurement", procurement.run)
graph.add_node("ranker", ranker.run)
graph.add_node("critic", critic.run)
graph.add_node("output", output.run)   # <-- your node

graph.set_entry_point("stylist")
graph.add_edge("stylist", "procurement")
graph.add_edge("procurement", "ranker")
graph.add_edge("ranker", "critic")
graph.add_edge("critic", "output")
graph.set_finish_point("output")
```

You don't need to write this — just make sure your `run()` function returns `{"output_text": ...}` and you're compatible with whatever state schema the team agrees on.

---

> Cross-agent coordination items for this agent are tracked in [`docs/eval_tasks_2026-04-20.md`](eval_tasks_2026-04-20.md) under **Cross-Agent Coordination Tasks**.

---

## What good output looks like

A strong response from your agent might look like:

> Your board reads as relaxed minimalism with an earth-tone palette — think undyed linen, washed indigo, and warm taupe, with a preference for loose, unconstructed silhouettes.
>
> **1. Uniqlo Wide Linen Trousers** — The texture and drape are a near-exact match for the linen pieces in your board. The taupe colorway lines up with 3 of your 7 reference images.
>
> **2. Mango Washed Denim Overshirt** — The worn-in finish and boxy cut fit your streetwear references. The critic flagged a similar jacket with visible branding, so this lower-logo option was ranked higher.
>
> 2 additional results were filtered for not meeting your vibe threshold — they matched keyword tags but had a sharper, more polished aesthetic than your board suggests.

---

## Dependencies to add to requirements.txt

```
anthropic>=0.28.0
```

LangGraph and the rest should already be in there from your teammates' work.
