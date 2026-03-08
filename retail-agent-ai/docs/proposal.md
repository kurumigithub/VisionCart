# INFO290: GenAI Final Project Proposal

## Group Members

1. Kurumi Kaneko (kurumi_kaneko@berkeley.edu)
2. Quinton King (quintonking@berkeley.edu)
3. Raras Pramudita (raras.pramudita@berkeley.edu)
4. Lance Santana (lance.santana@berkeley.edu)

## Github Repo Link

https://github.com/kurumigithub/VisionCart

---

## Project Description (400–500 words)

Modern e-commerce remains anchored to keyword-based retrieval, a modality that often fails to capture the aesthetic preferences of individual users. Consumers already curate rich visual data on platforms like Pinterest, Instagram, and Tiktok, yet these vision boards are disconnected from the point of purchase. This forces users to manually translate visual inspiration into text-based search queries, creating avoidable gaps between what they envision and what they can actually find. By leveraging Multimodal Models and Agentic reasoning, our system moves beyond simple item-matching toward true taste synthesis. Generative AI is uniquely suited for this task because it can encode high-dimensional aesthetic features, such as specific vibes or color palette of a modern street-style inspired space, that traditional search interfaces can’t define. This project builds a personalized, intent-driven retail agent that reduces search friction and filters out products misaligned with a user’s established taste profiles.

From a technical standpoint, the system is designed as an agentic AI architecture built on Multimodal Retrieval-Augmented Generation (RAG). The pipeline consists of three layers. First, a vision board encoder (e.g. GPT-4o or CLIP), will analyze a user’s curated images and generate a “style embedding” that captures textures, silhouettes, color palettes, and other preferences. This representation serves as the ground truth for the subsequent agentic discovery layer. Second, using LangChain, a reasoning agent (ReAct framework) translates this embedding into executable search queries and interacts with external APIs such as Google Shopping and Amazon Product Advertising. Third, the system performs a multimodal ranking step, where retrieved product images are passed through a CLIP-based similarity encoder to be ranked against the user’s original vision board embeddings.

Our approach significantly improves upon existing systems like Google Lens’ Style Search by shifting from reactive object-matching to proactive aesthetic synthesis. Rather than retrieving visually similar duplicates from a single image, our framework possesses a persistent “memory” of a user’s broader aesthetic history. As a secondary reach goal, we aim to implement a “critic” loop, an automated verification step where the LLM can discard a product that satisfies keyword matches, but fails to align with the user’s underlying stylistic ground truth, a level of semantic judgment that traditional search engines lack. Unlike Pinterest’s internal recommendations, our agentic layer also acts as a cross-platform bridge, translating high-level visual “vibes” into executable search queries across the open web.

To evaluate the success of this framework, we will utilize quantitative metrics common in recommendation systems, specifically Precision@K and Mean Reciprocal Rank (MRR). MRR will measure how high the agent ranks the stylistic matches within the top search results. Additionally, if time allows, we will conduct a qualitative “style alignment score” test, where the agent’s recommendations are compared against a baseline of standard keyword search results to determine which better reflects the user's original visual board. By integrating these GenAI principles, our project demonstrates a complete model lifecycle, from multimodal embedding representation to agentic tool-use and rigorous evaluation.

---

## Potential Data Sources

1. SerpAPI for Google Shopping  
   https://serpapi.com/google-shopping-api

2. Amazon Product Advertising API  
   https://webservices.amazon.com/paapi5/documentation/

3. Crawl from Pinterest dashboards

4. DeepFashion2 dataset  
   https://github.com/switchablenorms/DeepFashion2
   - If we decide to validate our style embeddings with a public dataset
