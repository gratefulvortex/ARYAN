# AI Document Query Assistant

## Project Structure
- `app.py`: Streamlit frontend
- `main.py`: FastAPI backend

## 1. Deploy the Backend (FastAPI)

### A. Deploy on Render (recommended)
1. Push your code to a GitHub repository.
2. Go to [https://render.com/](https://render.com/) and create a free account.
3. Click 'New' > 'Web Service'.
4. Connect your GitHub repo and select the repo containing `main.py`.
5. Set the build and start commands:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 8000`
6. Add environment variable `MISTRAL_API_KEY` (from your provider).
7. Deploy and wait for your public backend URL (e.g., `https://your-backend.onrender.com`).

## 2. Update Frontend API URLs
- In `app.py`, replace all `http://127.0.0.1:8000/` with your Render backend URL (e.g., `https://your-backend.onrender.com/`).

## 3. Deploy the Frontend (Streamlit Cloud)
1. Push your frontend code (`app.py`, `requirements.txt`) to a GitHub repository.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in.
3. Click 'New app', select your repo, and set the main file as `app.py`.
4. Click 'Deploy'.

## 4. Usage
- Upload Excel files via the Streamlit UI.
- Query your data using the chat interface.

---

**Note:** Both frontend and backend must be publicly accessible for the app to work. 