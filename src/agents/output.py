from transformers import pipeline

_pipe = None

def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-3B-Instruct",
            device_map="auto",
        )
    return _pipe


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
        tags = p.get("tags", [])
        tag_str = ", ".join(tags) if tags else "none"
        lines.append(
            f"{i}. {p.get('name', 'Unknown Product')}\n"
            f"   Score: {p.get('score', 'N/A')}\n"
            f"   Tags: {tag_str}\n"
            f"   Price: {p.get('price', 'N/A')}\n"
            f"   URL: {p.get('url', '')}"
        )
    return "\n\n".join(lines)


def build_output_products(ranked_products: list) -> list:
    """Trim ranked_products to the top 5 with only UI-relevant fields."""
    results = []
    for p in ranked_products[:5]:
        results.append({
            "name": p.get("name", "Unknown Product"),
            "price": p.get("price", "N/A"),
            "url": p.get("url", ""),
            "image_url": p.get("image_url", ""),
            "score": round(p.get("score"), 3) if p.get("score") is not None else None,
            "tags": p.get("tags", []),
        })
    return results


def run(state: dict) -> dict:
    """
    LangGraph node: generate human-readable output from ranked results.

    Reads from state:
        - style_profile (str): text description of the user's aesthetic
        - ranked_products (list): products sorted by similarity score
        - critic_notes (str, optional): any rejections or flags from the critic

    Writes to state:
        - output_text (str): narrative summary for the chat interface
        - output_products (list): top 3-5 products for UI card rendering
    """
    style_profile = state.get("style_profile", "")
    ranked_products = state.get("ranked_products", [])
    critic_notes = state.get("critic_notes", "")

    if not style_profile:
        return {
            "output_text": (
                "We couldn't determine your style profile. "
                "Try adding more images to your board."
            )
        }

    if not ranked_products:
        return {
            "output_text": (
                "We couldn't find products that matched your style profile. "
                "Try adding more images to your board."
            )
        }

    products_block = format_products_for_prompt(ranked_products)
    total = len(ranked_products)
    shown = min(total, 5)
    count_note = (
        f"(Showing top {shown} of {total} results)"
        if total > shown
        else f"({total} result{'s' if total != 1 else ''} found)"
    )

    user_message = f"""Style profile: {style_profile}

Ranked products {count_note}:
{products_block}
{f"{chr(10)}Critic notes: {critic_notes}" if critic_notes else ""}

Please generate the user-facing summary."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    pipe = get_pipe()
    result = pipe(messages, max_new_tokens=1024)
    output_text = result[0]["generated_text"][-1]["content"]
    output_products = build_output_products(ranked_products)
    return {"output_text": output_text, "output_products": output_products}
