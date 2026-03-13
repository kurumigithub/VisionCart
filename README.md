# VisionCart

**Group Members:** Kurumi Kaneko, Quinton King, Raras Pramudita, Lance Santana

## Overview

Modern e-commerce remains anchored to keyword-based retrieval, a modality that often fails to capture the aesthetic preferences of individual users. Consumers already curate rich visual data on platforms like Pinterest, Instagram, and Tiktok, yet these vision boards are disconnected from the point of purchase. This forces users to manually translate visual inspiration into text-based search queries, creating avoidable gaps between what they envision and what they can actually find. By leveraging Multimodal Models and Agentic reasoning, our system moves beyond simple item-matching toward true taste synthesis. This project builds a personalized, intent-driven retail agent that reduces search friction and filters out products misaligned with a user’s established taste profiles.

## Running the Project

1. **Clone the repository**:

   ```bash
   git clone https://github.com/kurumigithub/VisionCart.git
   cd VisionCart
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **NOT DONE YET**

### File Structure

```
VisionCart/
├── data/                  <- Sample vision boards, product images
├── docs/                  <- Project proposal, research papers, PM specs
├── src/
│   ├── agents/            <- Individual agent logic
│   │   ├── stylist.py     <- Qwen2.5-VL vision logic
│   │   ├── procurement.py <- Tool-calling & API logic
│   │   ├── ranker.py      <- clip / sigLIP?
│   │   ├── critic.py      <- Semantic "Vibe" verification
│   │   └── output.py      <- human-readable output
│   ├── tools/             <- API wrappers (Amazon, Google Shopping)
│   ├── graph/             <- LangGraph state & workflow definition
│   │   └── state.py       <- Shared state object
│   └── utils/             <- Local SigLIP ranking & embedding scripts
├── notebooks/             <- EDA and experimentation
├── tests/                 <- Evaluation scripts (Precision@K, MRR)
├── requirements.txt       <- Dependency management
└── README.md              <- Documentation & "Style Alignment Score" results
```

### Agent Structure
```
stylist          -> convert vision boards into a "style profile"
procurement      -> multi-step search across google/amazon apis
critic           -> filter out images semantically
ranker           -> mathematical rankings (cosine)
human-readable   -> synthesize all of the outputs
```
