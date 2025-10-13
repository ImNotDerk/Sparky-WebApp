# 🚀 Running the Application

To run this project locally, you need to start the backend and frontend in two separate terminals.

🖥️ Backend (Terminal 1)

```bash
cd backend
python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
echo > .env # place PROJECT_ID=<project_id, ENDPOINT_ID=<endpoint_id>, LOCATION=<location>
uvicorn main:app --reload
```

The backend will run at http://localhost:8000

🌐 Frontend (Terminal 2)

```bash
cd frontend
npm install
npm run dev
```

The frontend will run at http://localhost:3000
