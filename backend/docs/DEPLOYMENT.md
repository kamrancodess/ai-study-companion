# Free Deployment

## Local Run

```bash
cd ai_study_companion
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push `ai_study_companion/` to a GitHub repository.
2. Create a Streamlit Community Cloud app.
3. Set the app entry file to `app.py`.
4. Keep `data/`, uploads, local model caches, and `.venv/` out of Git.
5. Prefer smaller models because free hosting has memory limits.

## Render Free Tier

Use a web service with:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

For a zero-cost portfolio demo, Streamlit Community Cloud is usually simpler. For reliable production, plan for paid compute or a self-hosted machine because local transformer models need memory and startup time.

