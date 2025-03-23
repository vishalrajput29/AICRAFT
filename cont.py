import io
import streamlit as st
from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator
from fpdf import FPDF
import pandas as pd
from langchain.schema import HumanMessage
import plotly.express as px
import sqlite3
import random
import time
import re

try:
    GROQ_API_KEY = st.secrets["groq"]["api_key"]  # Get API key securely
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Mixtral-8x7b-32768", streaming=True)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# Initialize Google Translator (not used anymore but kept for future use)
translator = GoogleTranslator(source='auto', target='en')  # Default target language is English

# Function to generate PDF in memory
def generate_pdf_in_memory(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content.encode('latin-1', 'replace').decode('latin-1'))
    
    # Create a BytesIO buffer and write the PDF content into it
    pdf_output = io.BytesIO()
    pdf.output(pdf_output, dest='S')  # Use 'S' to write to a string buffer
    pdf_output.seek(0)   # Rewind the buffer to the beginning
    return pdf_output

# Database Initialization
def init_db():
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    
    # Create the users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE, 
            password TEXT NOT NULL
        )
    ''')
    
    # Create the progress table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            material_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create the questions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create the memory games table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            game_type TEXT NOT NULL,
            total_questions INTEGER NOT NULL,
            correct_answers INTEGER NOT NULL,
            wrong_answers INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Save Progress to Database
def save_progress(user_id, topic, material_type):
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO progress (user_id, topic, material_type)
        VALUES (?, ?, ?)
    ''', (user_id, topic, material_type))
    conn.commit()
    conn.close()

# Save Questions to Database
def save_question(user_id, question, answer):
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO questions (user_id, question, answer)
        VALUES (?, ?, ?)
    ''', (user_id, question, answer))
    conn.commit()
    conn.close()

# Save Memory Game Results to Database
def save_memory_game_results(user_id, game_type, total_questions, correct_answers, wrong_answers):
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO memory_games (user_id, game_type, total_questions, correct_answers, wrong_answers)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, game_type, total_questions, correct_answers, wrong_answers))
    conn.commit()
    conn.close()

# Retrieve Progress from Database
def get_progress(user_id):
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT topic, material_type, timestamp
        FROM progress
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Retrieve Questions from Database
def get_questions(user_id):
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT question, answer, timestamp
        FROM questions
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Retrieve Memory Game Results from Database
def get_memory_game_results(user_id):
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT game_type, total_questions, correct_answers, wrong_answers, timestamp
        FROM memory_games
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Login User
def login_user(email, password):
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    email = email.strip().lower()
    password = password.strip()
    cursor.execute('''
        SELECT id FROM users WHERE email = ? AND password = ?
    ''', (email, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        return user[0]  # Return user_id
    else:
        return None  # Invalid credentials

# Register User
def register_user(email, password):
    conn = sqlite3.connect('learnai.db')
    cursor = conn.cursor()
    try:
        email = email.strip().lower()
        password = password.strip()
        cursor.execute('''
            INSERT INTO users (email, password)
            VALUES (?, ?)
        ''', (email, password))
        conn.commit()
        conn.close()
        return True  # Registration successful
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Email already exists

# Initialize the database
init_db()

# Set Page Config
st.set_page_config(page_title="LearnAI+", page_icon="📚")
st.title("📚 LearnAI+: Personalized Learning Companion")

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Login/Register",
    "Ask Questions",
    "Generate Study Materials",
    "Document-Based Q&A",
    "Content Research & Writing",
    "Track Progress",
    "Memory Games"
])

# Page: Login/Register
if page == "Login/Register":
    st.header("Login or Register")
    login_or_register = st.selectbox("Choose an option", ["Login", "Register"])
    
    if login_or_register == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            user_id = login_user(email, password)
            if user_id:
                st.session_state.logged_in = True
                st.session_state.user_id = user_id
                st.success("Logged in successfully!")
            else:
                st.error("Invalid email or password.")
    
    elif login_or_register == "Register":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Register"):
            if register_user(email, password):
                st.success("Registered successfully! Please log in.")
            else:
                st.error("Email already exists.")

# Check if user is logged in
if not st.session_state.logged_in:
    st.warning("Please log in to access other features.")
else:
    # Initialize session state variables for logged-in users
    if "generated_material" not in st.session_state:
        st.session_state.generated_material = None
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "material_type" not in st.session_state:
        st.session_state.material_type = "Summary"
    if "math_score" not in st.session_state:
        st.session_state.math_score = 0
    if "word_recall_words" not in st.session_state:
        st.session_state.word_recall_words = []
    if "word_recall_score" not in st.session_state:
        st.session_state.word_recall_score = 0

    # Page: Ask Questions
    if page == "Ask Questions":
        st.header("Ask Your Questions")
        context = st.text_area("Provide Context (optional)", value="", height=100)
        question = st.text_input("Enter your question here:")
        
        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                input_text = f"{context}\n{question}" if context.strip() else question
                with st.spinner("Generating answer..."):
                    response = llm([HumanMessage(content=input_text)]).content
                    st.success("Answer:")
                    st.write(response)
                save_question(st.session_state.user_id, question, response)

    # Page: Generate Study Materials
    elif page == "Generate Study Materials":
        st.header("Generate Study Materials")
        st.session_state.topic = st.text_input(
            "Enter the topic you want to study:", value=st.session_state.topic
        )
        st.session_state.material_type = st.selectbox(
            "Select the type of material: ", [
                "Summary", "Quiz", "Flashcards", "Practice Problems", "Essay Outline",
                "Mind Map", "Vocabulary List", "Case Study", "Infographic Description",
                "Discussion Questions", "Project Ideas", "Mnemonics", "Timeline"
            ],
            index=[
                "Summary", "Quiz", "Flashcards", "Practice Problems", "Essay Outline",
                "Mind Map", "Vocabulary List", "Case Study", "Infographic Description",
                "Discussion Questions", "Project Ideas", "Mnemonics", "Timeline"
            ].index(st.session_state.material_type)
        )

        if st.button("Generate Material"):
            if st.session_state.topic.strip() == "":
                st.warning("Please enter a topic.")
            else:
                if st.session_state.material_type == "Summary":
                    prompt_text = f"Generate a concise summary about {st.session_state.topic}."
                elif st.session_state.material_type == "Quiz":
                    prompt_text = f"Create a quiz with 5 multiple-choice questions about {st.session_state.topic}."
                elif st.session_state.material_type == "Flashcards":
                    prompt_text = f"Create flashcards with key terms and definitions about {st.session_state.topic}."
                elif st.session_state.material_type == "Practice Problems":
                    prompt_text = f"Generate 5 practice problems related to {st.session_state.topic}."
                elif st.session_state.material_type == "Essay Outline":
                    prompt_text = f"Create an essay outline about {st.session_state.topic}."
                elif st.session_state.material_type == "Mind Map":
                    prompt_text = f"Create a mind map for {st.session_state.topic}."
                elif st.session_state.material_type == "Vocabulary List":
                    prompt_text = f"Create a vocabulary list with definitions for {st.session_state.topic}."
                elif st.session_state.material_type == "Case Study":
                    prompt_text = f"Generate a case study about {st.session_state.topic}."
                elif st.session_state.material_type == "Infographic Description":
                    prompt_text = f"Describe an infographic for {st.session_state.topic}."
                elif st.session_state.material_type == "Discussion Questions":
                    prompt_text = f"Generate discussion questions about {st.session_state.topic}."
                elif st.session_state.material_type == "Project Ideas":
                    prompt_text = f"Suggest project ideas for {st.session_state.topic}."
                elif st.session_state.material_type == "Mnemonics":
                    prompt_text = f"Create mnemonics to help remember concepts related to {st.session_state.topic}."
                elif st.session_state.material_type == "Timeline":
                    prompt_text = f"Create a timeline of key events or processes related to {st.session_state.topic}."

                with st.spinner("Generating material..."):
                    response = llm([HumanMessage(content=prompt_text)]).content
                    st.session_state.generated_material = response
                    st.success(f"Generated {st.session_state.material_type}:")
                    st.write(st.session_state.generated_material)
                save_progress(st.session_state.user_id, st.session_state.topic, st.session_state.material_type)

    # Page: Document-Based Q&A
    elif page == "Document-Based Q&A":
        st.header("📚 Document-Based Q&A")
        st.write("Upload a document and ask questions based on its content.")
        uploaded_file = st.file_uploader("Upload a file (PDF, Excel, CSV, TXT)", type=["pdf", "xlsx", "csv", "txt"])

        if uploaded_file:
            file_type = uploaded_file.type
            st.success(f"File uploaded successfully: {uploaded_file.name} ({file_type})")

            if file_type == "application/pdf":
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    text = "\n".join(page.extract_text() for page in pdf.pages)
                st.write("Extracted text from PDF:")
                st.text_area("Preview", value=text[:500] + "...", height=150)

            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":  # Excel  
                df = pd.read_excel(uploaded_file)
                text = df.to_string()
                st.write("Extracted data from Excel:")
                st.dataframe(df)

            elif file_type == "text/csv":  # CSV
                df = pd.read_csv(uploaded_file)
                text = df.to_string()
                st.write("Extracted data from CSV:")
                st.dataframe(df)

            elif file_type == "text/plain":  # TXT
                text = uploaded_file.getvalue().decode("utf-8")
                st.write("Extracted text from TXT:")
                st.text_area("Preview", value=text[:500] + "...", height=150)

            else:
                st.error("Unsupported file type.")
                text = None

            if text:
                question = st.text_input("Ask a question about the document: ")
                if st.button("Get Answer"):
                    if question.strip() == "":
                        st.warning("Please enter a question.")
                    else:
                        from langchain.text_splitter import RecursiveCharacterTextSplitter
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=2000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(text)

                        def find_relevant_chunk(chunks, question):
                            keywords = set(question.lower().split())
                            relevance_scores = [
                                sum(keyword in chunk.lower() for keyword in keywords)
                                for chunk in chunks
                            ]
                            most_relevant_index = relevance_scores.index(max(relevance_scores))
                            return chunks[most_relevant_index]

                        relevant_chunk = find_relevant_chunk(chunks, question)
                        input_text = f"Document Content:\n{relevant_chunk}\n\nQuestion: {question}"
                        response = llm([HumanMessage(content=input_text)]).content
                        st.success("Answer:")
                        st.write(response)
                        save_question(st.session_state.user_id, question, response)
                        
    # Page: Content Research & Writing
    elif page == "Content Research & Writing":
        st.header("📝 Content Research & Writing")
        st.subheader("Generate Curated Content and Find Relevant YouTube Channels")

        topic = st.text_input("Enter the topic for content research:")
        if st.button("Generate Content"):
            if topic.strip() == "":
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating content..."):
                    # Generate curated content
                    content_prompt = f"Research and write a detailed article about {topic}, including key points, examples, and references."
                    content_response = llm([HumanMessage(content=content_prompt)]).content
                    st.success("Generated Content:")
                    st.write(content_response)

                    # Find relevant YouTube channels
                    youtube_prompt = f"Suggest 5 popular YouTube channels that create content about {topic}."
                    youtube_response = llm([HumanMessage(content=youtube_prompt)]).content
                    st.success("Relevant YouTube Channels:")
                    st.write(youtube_response)

                    # Save progress
                    save_progress(st.session_state.user_id, topic, "Content Research & Writing")
                    
    # Page: Track Progress
    elif page == "Track Progress":
        st.header("Track Your Progress")
        progress_data = get_progress(st.session_state.user_id)
        questions_data = get_questions(st.session_state.user_id)
        memory_game_data = get_memory_game_results(st.session_state.user_id)

        if not progress_data and not questions_data and not memory_game_data:
            st.info("No progress, questions, or memory game data available yet.")
        else:
            if progress_data:
                st.subheader("Your Learning Activities")
                df_progress = pd.DataFrame(progress_data, columns=["Topic", "Material Type", "Timestamp"])
                st.dataframe(df_progress)

                st.subheader("Material Types Distribution")
                material_counts = df_progress["Material Type"].value_counts().reset_index()
                material_counts.columns = ["Material Type", "Count"]
                fig_materials = px.bar(material_counts, x="Material Type", y="Count", title="Material Types Distribution")
                st.plotly_chart(fig_materials)

            if questions_data:
                st.subheader("Your Questions and Answers")
                df_questions = pd.DataFrame(questions_data, columns=["Question", "Answer", "Timestamp"])
                st.dataframe(df_questions)

                df_questions["Date"] = pd.to_datetime(df_questions["Timestamp"]).dt.date
                df_questions["DayOfWeek"] = pd.to_datetime(df_questions["Timestamp"]).dt.day_name()
                df_questions["Hour"] = pd.to_datetime(df_questions["Timestamp"]).dt.hour

                st.subheader("Questions Over Time")
                questions_over_time = df_questions.groupby("Date").size().reset_index(name="Questions Count")
                fig_questions = px.line(
                    questions_over_time,
                    x="Date",
                    y="Questions Count",
                    title="Questions Asked Over Time",
                    markers=True
                )
                st.plotly_chart(fig_questions)

                st.subheader("Questions by Day of the Week")
                questions_by_day = df_questions["DayOfWeek"].value_counts().reset_index()
                questions_by_day.columns = ["DayOfWeek", "Questions Count"]
                fig_days = px.bar(
                    questions_by_day,
                    x="DayOfWeek",
                    y="Questions Count",
                    title="Questions Asked by Day of the Week"
                )
                st.plotly_chart(fig_days)

            if memory_game_data:
                st.subheader("Memory Game Performance")
                df_memory = pd.DataFrame(memory_game_data, columns=["Game Type", "Total Questions", "Correct Answers", "Wrong Answers", "Timestamp"])
                st.dataframe(df_memory)

                df_memory["Date"] = pd.to_datetime(df_memory["Timestamp"]).dt.date
                aggregated_data = df_memory.groupby("Date").agg({
                    "Total Questions": "sum",
                    "Correct Answers": "sum",
                    "Wrong Answers": "sum"
                }).reset_index()

                st.subheader("Total Questions Solved Over Time")
                fig_total = px.line(
                    aggregated_data,
                    x="Date",
                    y="Total Questions",
                    title="Total Questions Solved Over Time",
                    markers=True
                )
                st.plotly_chart(fig_total)

                st.subheader("Correct vs Wrong Answers")
                fig_correct_wrong = px.bar(
                    aggregated_data,
                    x="Date",
                    y=["Correct Answers", "Wrong Answers"],
                    title="Correct vs Wrong Answers",
                    barmode="stack"
                )
                st.plotly_chart(fig_correct_wrong)

    # Page: Memory Games
    elif page == "Memory Games":
        st.header("🧠 Memory Games")
        st.subheader("Boost Your Memory with Fun Games!")

        if "math_problem" not in st.session_state:
            st.session_state.math_problem = None
            st.session_state.math_answer = None

        if st.button("Start Math Puzzle"):
            num1 = random.randint(1, 100)
            num2 = random.randint(1, 100)
            operation = random.choice(["+", "-", "*", "/"])

            if operation == "+":
                result = num1 + num2
            elif operation == "-":
                result = num1 - num2
            elif operation == "*":
                result = num1 * num2
            elif operation == "/":
                num1 = num1 * num2
                result = num1 / num2

            st.session_state.math_problem = f"{num1} {operation} {num2}"
            st.session_state.math_answer = result
            st.success(f"Solve this: {st.session_state.math_problem}")

        if st.session_state.math_problem:
            user_answer = st.text_input("Your Answer:") 
            if st.button("Submit Answer"):
                try:
                    user_answer = float(user_answer)
                    if user_answer == st.session_state.math_answer:
                        st.session_state.math_score += 1
                        st.success(f"Correct! Your score: {st.session_state.math_score}")
                        save_memory_game_results(st.session_state.user_id, "Math Puzzle", 1, 1, 0)
                    else:
                        st.error(f"Wrong! The correct answer was {st.session_state.math_answer}.")
                        save_memory_game_results(st.session_state.user_id, "Math Puzzle", 1, 0, 1)
                except ValueError:
                    st.error("Please enter a valid number.")

        st.markdown("### 📝 Word Recall Game")
        if "word_recall_stage" not in st.session_state:
            st.session_state.word_recall_stage = "display"
            st.session_state.word_recall_words = []

        if st.session_state.word_recall_stage == "display":
            st.write("Memorize the following words:")
            words = [random.choice(["apple", "banana", "cat", "dog", "elephant", "frog", "giraffe"]) for _ in range(5)]
            st.session_state.word_recall_words = words
            st.success(", ".join(words))
            time.sleep(5)
            st.session_state.word_recall_stage = "recall"

        elif st.session_state.word_recall_stage == "recall":
            st.write("Now type the words you remember:")
            recalled_words = st.text_input("Words (separated by commas):")
            if st.button("Check Words"):
                recalled_words = [word.strip() for word in recalled_words.split(",")]
                correct_words = set(recalled_words).intersection(st.session_state.word_recall_words)
                st.session_state.word_recall_score += len(correct_words)
                wrong_words = len(st.session_state.word_recall_words) - len(correct_words)
                st.success(f"You remembered {len(correct_words)} out of {len(st.session_state.word_recall_words)} words. Score: {st.session_state.word_recall_score}")
                save_memory_game_results(st.session_state.user_id, "Word Recall", len(st.session_state.word_recall_words), len(correct_words), wrong_words)
                st.session_state.word_recall_stage = "display"



# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built by ❤️ **TEAM OUTLIERS**")
