# **AniSuggest: AI-Powered Anime Recommendation Engine**

**AniSuggest** is an intelligent web platform designed to provide anime fans with highly personalized and semantically relevant recommendations. Moving beyond simple genre-matching, AniSuggest uses modern Natural Language Processing (NLP) to understand the core themes and narrative of an anime's synopsis, allowing it to uncover hidden gems and suggest content that truly resonates with a user's specific tastes.

**Live Demo:** [**https://ani-sugg.vercel.app/**](https://www.google.com/search?q=https://ani-sugg.vercel.app/)

## **✨ Features**

* **Semantic Search:** Find anime based on the meaning and themes of their stories, not just keywords.  
* **Content-Based Recommendations:** Get a list of similar shows based on any anime you're interested in ("Because you viewed...").  
* **Personalized "For You" Page:** The app analyzes your entire watched history to create a unique "taste profile," generating recommendations tailored specifically to you.  
* **Dynamic Filtering:** Already-watched anime are automatically filtered out from all recommendation and browsing lists.  
* **Interactive UI:** A sleek, modern, and responsive user interface built with React and Tailwind CSS.

## **🛠️ Tech Stack**

This project is a full-stack application built with a modern technology stack, as outlined in the initial PRD.

| Component | Technology | Rationale |
| :---- | :---- | :---- |
| **Frontend** | React (Vite) & TypeScript | For a fast, modern, and type-safe user interface. |
| **Backend** | Python & FastAPI | High-performance API framework within a robust AI/ML ecosystem. |
| **Database** | MongoDB | Flexible NoSQL database for storing user profiles and watched lists. |
| **AI Model** | sentence-transformers | State-of-the-art library for generating high-quality vector embeddings from text. |
| **Deployment** | **Vercel** (Frontend), **Render** (Backend), **MongoDB Atlas** (Database) | A serverless, scalable, and cost-effective cloud infrastructure. |

## **🚀 Getting Started**

To get a local copy up and running, follow these simple steps.

### **Prerequisites**

Make sure you have the following software installed on your machine:

* [Git](https://git-scm.com/)  
* [Python 3.11+](https://www.python.org/downloads/)  
* [Node.js and npm](https://nodejs.org/en)  
* [MongoDB Community Server](https://www.mongodb.com/try/download/community) (ensure the service is running)

### **Installation & Setup**

1. **Clone the repository:**  
   git clone \[https://github.com/YourUsername/anisugg.git\](https://github.com/YourUsername/anisugg.git)  
   cd anisugg

2. **Setup the Backend:**  
   * Navigate to the backend directory:  
     cd backend

   * Create and activate a Python virtual environment:  
     python \-m venv .venv  
     \# On Windows  
     .\\.venv\\Scripts\\activate  
     \# On macOS/Linux  
     source .venv/bin/activate

   * Install the required Python packages:  
     pip install \-r requirements.txt

   * **Crucial Step:** Run the data processing script to fetch the anime data and generate the AI model files. This will create a data folder.  
     python scripts/process\_data.py

   * Start the backend server. It will run on http://127.0.0.1:8000.  
     uvicorn main:app \--reload

3. **Setup the Frontend:**  
   * Open a **new terminal** and navigate to the frontend directory:  
     cd frontend

   * Install the required npm packages:  
     npm install

   * Run the frontend development server. It will be available at http://localhost:5173.  
     npm run dev

You should now have the full application running locally\!

## **☁️ Deployment**

The live version of this application is deployed using a modern cloud stack:

* The **React frontend** is hosted on **Vercel**, which automatically deploys on every push to the main branch.  
* The **FastAPI backend** is hosted as a Web Service on **Render**, which also deploys automatically.  
* The **MongoDB database** is a managed cluster hosted on **MongoDB Atlas**.

The MONGO\_URI and VITE\_API\_URL are configured as environment variables in their respective cloud services for security and flexibility.

