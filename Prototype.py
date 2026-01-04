import streamlit as st
import pandas as pd
import time
import subprocess
import uuid
import os
import torch
import shutil
import requests

from io import BytesIO
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


# --- MODEL CACHING ---
@st.cache_resource(show_spinner="Loading AI Art Model (Run Once)...")
def get_pipeline():
    print(">>> INITIALIZING AI MODEL... <<<")

    base_model = "cagliostrolab/animagine-xl-3.1"
    ip_adapter_repo = "h94/IP-Adapter"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        add_watermarker=False
    )

    # 2. Load Adapter
    pipe.load_ip_adapter(
        ip_adapter_repo,
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin"
    )

    # 3. Optimize Memory
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()

    # 4. Set Scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="sde-dpmsolver++"
    )

    return pipe


def generate_story_image(prompt: str, portrait_path: str | None = None) -> Image.Image:
    """
    Generates an image using the cached pipeline.
    """
    # Retrieve the model from cache (Instant!)
    pipe = get_pipeline()

    kwargs = {
        "prompt": prompt,
        "guidance_scale": 7.0,
        "num_inference_steps": 30,
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, realistic, photo, 3d render, lips, nose, fused bodies, fused heads, multiple boys, multiple girls, mutation"
    }

    if portrait_path:
        # Resize input to 1024x1024 for best SDXL performance
        try:
            ref_img = Image.open(portrait_path).convert("RGB").resize((1024, 1024))
            kwargs["ip_adapter_image"] = ref_img
            pipe.set_ip_adapter_scale(0.6)
        except Exception as e:
            print(f"Error loading portrait: {e}")

    # Generate
    result = pipe(**kwargs)
    return result.images[0]

# --- END OF AI CODE ---

st.set_page_config(page_title="Social Stories Prototype", layout="wide")

PORTRAITS_DIR = Path("uploads/portraits")
PORTRAITS_DIR.mkdir(parents=True, exist_ok=True)

STORY_IMAGES_DIR = Path("uploads/story_images")
STORY_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# --- 0. CUSTOM CSS STYLING ---
st.markdown("""
<style>
/* Hide menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* App background */
.stApp {
    background-color: #f7f3e8;
}

/* Unified TextInput + Selectbox styling */
.stTextInput > div > div > input,
div[data-testid="stSelectbox"] > div > div {
    background-color: #fce7db !important;
    border: 1px solid #d4a796 !important;
    border-radius: 15px !important;
    color: #444 !important;
    font-weight: bold;
    font-size: 16px;
}

/* TextArea - UPDATED TO MATCH INPUTS */
div.stTextArea textarea {
    background-color: #fce7db !important;
    border: 1px solid #d4a796 !important;
    border-radius: 15px !important;
    color: #444 !important;
    font-weight: bold !important;
    font-size: 16px !important;
}

/* Buttons (default + primary) */
div.stButton > button {
    border-radius: 15px;
    padding: 10px;
    font-size: 16px;
    font-weight: bold;
    color: #444;
    background-color: #fce7db;
    border: 1px solid #d4a796;
}

div.stButton > button:first-child,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #e5987d !important;
    color: white !important;
    border: none !important;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
    transition: all 0.2s;
}

/* Hover */
div.stButton > button:first-child:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #d1876e !important;
}

/* Titles */
h1 {
    font-size: 3.5em !important;
    font-weight: 700;
    color: #333;
}
h3 {
    color: #666;
}

</style>
""", unsafe_allow_html=True)


if "llm_questions" not in st.session_state:
    st.session_state.llm_questions = []

if "llm_answers" not in st.session_state:
    st.session_state.llm_answers = []

if "question_index" not in st.session_state:
    st.session_state.question_index = 0


with st.sidebar:
    st.markdown("### Dev Navigation")
    # st.title("Menu")
    # st.write(f"Logged in as Admin")
    if st.button("Go to login"):
        st.session_state.page = "login"
        st.rerun()
    if st.button("Go to Manage_Profiles"):
        st.session_state.page = "Manage_Profiles"
        st.rerun()
    if st.button("Go to new_profile"):
        st.session_state.page = "new_profile"
        st.rerun()
    if st.button("Go to child_profile"):
        st.session_state.page = "child_profile"
        st.rerun()
    if st.button("Go to create_new_story"):
        st.session_state.page = "create_new_story"
        st.rerun()
    if st.button("Go to new_school_story"):
        st.session_state.page = "new_school_story"
        st.rerun()
    if st.button("Go to new_pet_story"):
        st.session_state.page = "new_pet_story"
        st.rerun()
    if st.button("Go to new_baby_story"):
        st.session_state.page = "new_baby_story"
        st.rerun()
    if st.button("Go to custom_story_details"):
        st.session_state.page = "custom_story_details"
        st.rerun()
    if st.button("Go to additional_details_1"):
        st.session_state.page = "additional_details_1"
        st.rerun()
    if st.button("Go to additional_details_2"):
        st.session_state.page = "additional_details_2"
        st.rerun()
    if st.button("Go to story_display"):
        st.session_state.page = "story_display"
        st.rerun()
    if st.button("Go to like_story"):
        st.session_state.page = "like_story"
        st.rerun()
    if st.button("Go to dislike_story"):
        st.session_state.page = "dislike_story"
        st.rerun()
    if st.button("Go to saved_story_display"):
        st.session_state.page = "saved_story_display"
        st.rerun()



# Initialize session state for navigation and data persistence
if 'page' not in st.session_state:
    st.session_state.page = 'login'

if 'data' not in st.session_state:
    # Mock Data structure for the prototype's "database"
    st.session_state.data = pd.DataFrame({
        'ID': [],
        'Full Name': [],
        'Age': [],
        'Gender': [],
        'Status': []
    })


# --- Initialize Saved Stories Data ---
if 'saved_stories' not in st.session_state:
    st.session_state.saved_stories = []

# Navigation helper function
def navigate_to(page_name):
    """Updates the session state to switch the active page view."""
    st.session_state.page = page_name


# Communication with LLM helper functions
# LLM Calling
# def ask_ollama(prompt, model="llama3"):
#     OLLAMA_PATH = r"C:\Users\sahar\AppData\Local\Programs\Ollama\ollama.exe"
#
#     # 1. EXISTING: Generate the story using your current CLI method
#     result = subprocess.run(
#         [OLLAMA_PATH, "run", model, prompt],
#         capture_output=True,
#         text=True
#     )
#
#     # 2. NEW: Force Unload from GPU immediately
#     # We do this inside a try-block so it doesn't crash your app if something goes wrong
#     try:
#         import requests
#         requests.post(
#             "http://localhost:11434/api/generate",
#             json={"model": model, "keep_alive": 0}
#         )
#         print(">>> OLLAMA UNLOADED (VRAM CLEARED) <<<")
#     except Exception as e:
#         print(f"Could not unload Ollama: {e}")
#
#     return result.stdout.strip()

def ask_ollama(prompt, model="mistral"):
    OLLAMA_PATH = r"C:\Users\sahar\AppData\Local\Programs\Ollama\ollama.exe"

    # 1. GENERATE
    result = subprocess.run(
        [OLLAMA_PATH, "run", model, prompt],
        capture_output=True,
        text=True
    )

    # 2. FORCE UNLOAD
    try:
        import requests
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "keep_alive": 0}
        )
    except Exception:
        pass

    # 3. ROBUST CLEANUP
    raw_text = result.stdout.strip()

    # List of phrases to cut off
    stop_phrases = [
        # "ratio:",
        # "note:",
        # "(note:",
        # "( note:",
        # "please note:",
        "**end of story",
        "end of story",
        # "this social story follows",
        # "i hope this helps",
        # "let me know if",
        # "--",
    ]

    # Loop through and cut if found (Case Insensitive)
    for phrase in stop_phrases:
        # Check if the LOWERCASE version of the phrase is in the LOWERCASE text
        if phrase in raw_text.lower():
            # Find the actual index where it starts (ignoring case)
            start_index = raw_text.lower().find(phrase)

            # Print notification
            print(f"\n CLEANUP ACTION: Detected '{phrase}' - Removing extra text automatically.\n")

            # Cut the text at that index
            raw_text = raw_text[:start_index].strip()

    return raw_text


if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""


# LLM Prompt
def get_child_profile_str():
    """Helper: Formats selected child data into a string for the LLM"""
    if 'selected_child' not in st.session_state:
        return "Child profile is unknown."

    c = st.session_state.selected_child

    profile = f"""
    CHILD PROFILE:
    - Name: {c.get('Full Name', 'Unknown')}
    - Age: {c.get('Age', 'Unknown')}
    - Gender: {c.get('Gender', 'Unknown')}
    - Functional Level: {c.get('Functional Level', 'Unknown')}
    - Communication/Reading: {c.get('Reading Level', 'Unknown')}
    - Interests: {c.get('Interests', 'None listed')}
    - Sensitivities/Sensory: {c.get('Sensory Needs', 'None listed')}
    - Strengths: {c.get('Strengths', 'None listed')}
    - General Details: {c.get('General Details', 'None listed')}
    """
    return profile

def build_question_prompt(context):
    """
    Generates a prompt for the LLM to ask the next clarifying question.
    Uses the '6 Wh- Questions' framework to identify gaps.
    """
    return f"""
You are an expert consultant helping create a Social Story for children.
Your goal is to gather the missing information needed to write an effective story.
Use the Child Profile below to adapt your tone and complexity.

YOUR INTERNAL MENTAL GUIDE - You may use this 6 dimensions to make sure you have the
information you need, or what is missing:
1. WHO is involved?
2. WHERE does it happen?
3. WHEN does it happen?
4. WHAT exactly happens?
5. WHY is it challenging?
6. HOW should the child respond?

Use the Situation Details to understand the context.
Ask ONE short, simple, clear follow-up question to fill ONE gap or to better personalize the story.


CONTEXT AND DATA:
{context}

Only output the question. No explanations.
"""

def build_image_prompt(story_text, child_profile, has_portrait: bool):
    portrait_instruction = (
        "The main child in the illustration should closely resemble the provided reference image. "
        "Do not invent facial features that conflict with the reference. "
        "Maintain a gentle, child-friendly appearance."
        if has_portrait
        else ""
    )

    return f"""
You are generating a visual illustration for a children's Social Story.

RULES:
- Describe ONE calm, child-friendly scene
- Illustration style (storybook / animated), not realistic photo
- No text, no captions, no speech bubbles
- Focus on emotions, setting, and relationships
- The child should appear age-appropriate and gentle

{portrait_instruction}

CHILD PROFILE:
{child_profile}

STORY:
{story_text}

OUTPUT:
Write a short visual prompt - ONE sentence only.
Do NOT exceed this limit.
Only output the prompt text.
"""

def build_new_school_context():
    """Helper: Formats the specific new school story details"""

    school_type = st.session_state.get("school_type", "").strip()
    school_name = st.session_state.get("school_name", "").strip()
    notes = st.session_state.get("school_additional_notes", "").strip()

    return (
        "STORY TYPE: Starting a New School\n"
        f"Type of school: {school_type}\n"
        f"School name: {school_name}\n"
        f"Specific details/notes: {notes}\n"
    )

def build_new_pet_context():
    """Helper: Formats the specific pet story details"""
    # Get data safely from session state
    pet_type = st.session_state.get("pet_type", "").strip()
    pet_name = st.session_state.get("pet_name", "").strip()
    notes = st.session_state.get("pet_additional_notes", "").strip()

    # Build text for LLM
    return (
        "STORY TYPE: Getting a New Pet\n"
        f"Type of animal: {pet_type}\n"
        f"Pet's name: {pet_name}\n"
        f"Specific details/notes: {notes}\n"
    )

def build_new_sibling_context():
    """Helper: Formats the specific new sibling story details"""

    baby_gender = st.session_state.get("baby_gender", "").strip()
    notes = st.session_state.get("baby_additional_notes", "").strip()

    return (
        "STORY TYPE: Having a New Baby Sibling\n"
        f"Baby's gender: {baby_gender}\n"
        f"Specific details/notes: {notes}\n"
    )

# --- CONSTANT: THE MASTER SYSTEM PROMPT ---
GUIDELINES_FOR_WRITING = f""""
1. Main Character: 
The child (from the profile) MUST be the main character.
Personalize the story so the child feels represented and comfortable.
2. Setting and Characters
Be specific about all settings in the story.
Clearly describe other characters and their roles.
3. Dialogue
Include realistic, age-appropriate and ability-appropriate dialogue.
Use first-person (“I/we”) and/or third-person (“he/she/they”) perspective.
Avoid “you” statements;
4. Story Structure
Include a clear title, beginning, middle, and end.
Repeat key points to reinforce understanding.
5. Tone and Vocabulary
Keep a positive, patient, and non-judgmental tone.
Use literal, accurate language; avoid metaphors, idioms, or ambiguous terms.
Prefer constructive statements.
6. Child Considerations
Tailor the story to the child’s cognitive abilities: Learning style, Attention span, Comprehension level, Reading ability.
7. Content Accuracy
Ensure information is factual and meaningful.
Be precise with verbs and actions.
Cover the who, what, where, when, why, and how of the situation.
8. Goal
Each story should have one clear goal.
The story should focus on teaching social understanding rather than directly addressing problem behavior.
The story should teach context, not just give orders.
9. Personalization / Interests
Incorporate the child’s interests to make the story engaging.
Avoid rigid, inflexible language; use qualifiers like “usually,” “sometimes,” or “probably.”
10. Engagement Features
Stories may include repetition of important points to reinforce learning.
"""

SENTENCE_TYPES_RATIO = f""""
1. Descriptive Sentences
Describe the child/adolescent, the environment, and what will happen in the situation.
Present clear, factual, objective information.
Purpose: set context and prepare the child for the situation.
2. Directive / Coaching Sentences
Gently guide the child on how to respond in the situation.
Types:
a. Child-directed: Suggest possible responses.
b. Caregiver-directed: Suggest what adults can do to help.
c. Child-chosen: The child decides their own strategy.
3. Perspective Sentences
Describe the thoughts, feelings, or reactions of others in the situation.
Helps develop social understanding and empathy.
4. Cooperative Sentences
Explain how others will support or assist the child.
5. Affirmative Sentences
Reinforce other sentences or social rules.
Highlight values or positive outcomes.
6. Ratio
Use the ratio of one directive sentence to two or more sentences of the other types in every Social Story.

IMPORTANT: This is a silent set of rules for your internal calculation only. NEVER output the ratio score or explain the sentence types!
"""

SOCIAL_STORY_MASTER_PROMPT = f"""
You are a world-renowned expert in Special Education and Speech-Language Pathology, specializing in creating Social Stories based on Carol Gray's methodology.
Your goal is to help a child navigate a new social situation safely and confidently.

### STEP 1: ANALYZE THE CHILD PROFILE
Before writing, look at the provided "Child Profile" data.
Use all provided details to personalize the story.
- Cognitive Level: Adapt your vocabulary and sentence complexity to the child's Age, Functional Level, comprehension, attention span, and Reading Level.
- Interests: Incorporate the child's specific interests, motivators and strengths to increase engagement.
- Sensory: Be mindful of the environment described; avoid triggering language and acknowledge potential sensory challenges if relevant.

### STEP 2: GUIDELINES FOR WRITING (CAROL GRAY METHOD)
{GUIDELINES_FOR_WRITING}

### STEP 3: SENTENCE TYPES & RATIO
You must use a mix of these sentence types:
{SENTENCE_TYPES_RATIO}

### STEP 4: OUTPUT FORMAT
- The VERY FIRST line must be the Title of the story. Do not write "Title:", just the title text.
- Leave one blank line.
- Then start the story content.
- The VERY LAST line must be "end of story".
- No explanations or definitions except the story itself.
No Intro: Do NOT write "Here is the story". Start directly with the Title.
"""


def build_final_story_prompt(context):
    return f"""
You are an expert Social Story creator designed to help children navigate social situations.

--- MASTER GUIDELINES (ALWAYS FOLLOW THESE) ---
{SOCIAL_STORY_MASTER_PROMPT}

--- SPECIFIC STORY DATA ---
The following is the specific information about the child and the situation:
{context}

--- TASK ---
Based on the Master Guidelines and the Specific Story Data above, write a complete Social Story now.
"""

def build_regeneration_full_prompt(feedback):
    """Helper: Gathers all data and builds the prompt for regenerating a story."""

    # 1. Gather all existing data from session state
    child_profile = get_child_profile_str()
    situation = st.session_state.get("initial_story_context", "")

    qa_history = ""
    for q, a in zip(st.session_state.llm_questions, st.session_state.llm_answers):
        qa_history += f"Q: {q}\nA: {a}\n"

    # 2. Build the specific context string including the feedback
    context_with_feedback = f"""
    Child Profile:
    {child_profile}

    ORIGINAL SITUATION:
    {situation}

    ADDITIONAL DETAILS (Q&A):
    {qa_history}

    --- FEEDBACK & INSTRUCTIONS ---
    The user was not satisfied with the previous version.
    Rewrite the story to address this feedback:
    "{feedback}"

    --- CRITICAL FORMATTING RULE ---
    OUTPUT ONLY THE STORY. 
    DO NOT WRITE "Here is the rewritten story". 
    START DIRECTLY WITH THE TITLE.
    """

    # 3. Wrap it in the Master Guidelines
    return build_final_story_prompt(context_with_feedback)


def generate_story_from_session():
    """
    Central Engine: Gathers Profile + Context + Q&A and calls the LLM.
    Returns: The generated story text.
    """
    # 1. Get Child Profile
    child_profile = get_child_profile_str()

    # 2. Get Situation (Context)
    situation_details = st.session_state.get("initial_story_context", "")

    # 3. Build Q&A History (if exists)
    qa_context = ""
    if "llm_questions" in st.session_state and "llm_answers" in st.session_state:
        for q, a in zip(st.session_state.llm_questions, st.session_state.llm_answers):
            qa_context += f"Q: {q}\nA: {a}\n"

    # 4. Combine EVERYTHING
    final_full_context = f"""
    --- Child Profile ---
    {child_profile}

    --- SITUATION DETAILS ---
    {situation_details}

    --- ADDITIONAL DETAILS (Q&A) ---
    {qa_context}
    """

    # 5. Build Prompt & Call LLM
    final_prompt = build_final_story_prompt(final_full_context)
    return ask_ollama(final_prompt)


# --- 2. Screen Definitions ---

def show_login():
    """Screen 1: Login Page"""

    col_img_left, col_content, col_img_right = st.columns([2.0, 2.5, 1.0])

    with col_img_left:
        st.markdown("<br>" * 7, unsafe_allow_html=True)

        try:
            st.image("images/mother_child.png", width="stretch")
        except:
            st.warning("Could not find 'images/mother_child.png'. Please ensure the 'images' folder exists.")

        st.markdown(
            """
            <p style='margin-top: 5px; font-size: 16px; text-align: center;'>Based on Carol Gray's theoretical framework</p>
            """,
            unsafe_allow_html=True
        )

    with col_content:
        st.markdown("<h1 style='text-align: center;'>Social Stories</h1>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align: center;'>Create Meaningful Personalized Social Stories for Your Child!</h3>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", label_visibility="collapsed", placeholder="Username")
            password = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")

            submit = st.form_submit_button("Sign In", type="primary", width="stretch")
            st.markdown("<div style='text-align: center;'><a href='#'>Forgot password?</a></div>",
                        unsafe_allow_html=True)

            if submit:
                # Login logic for prototype: accepts "admin/1234" or any username
                if username == "admin" and password == "1234":
                    st.success(f"Welcome back, {username}!")
                    time.sleep(1)
                    navigate_to('Manage_Profiles')
                    st.rerun()
                elif username:
                    st.success(f"Welcome, {username}!")
                    time.sleep(1)
                    navigate_to('Manage_Profiles')
                    st.rerun()
                else:
                    st.error("Please enter a username.")

        st.button("Sign Up", width="stretch")
        st.markdown("<div style='text-align: center; margin-top: 20px;'><a href='#'>Contact Us</a></div>",
                    unsafe_allow_html=True)

    with col_img_right:
        st.markdown("<br>" * 18, unsafe_allow_html=True)
        try:
            st.image("images/books_stack.png", width="stretch")
        except:
            st.warning("Could not find 'images/books_stack.png'. Please ensure the 'images' folder exists.")

    # # --- SUBTITLE WITH QUESTION MARK ---
    # sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])
    #
    # with sub_col1:
    #     # --- Help Icon ---
    #     st.markdown(
    #         '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
    #         unsafe_allow_html=True
    #     )

    # --- SUBTITLE WITH QUESTION MARK ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])

    with sub_col1:
        # --- CSS for Tooltip ---
        st.markdown("""
        <style>
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }

        /* The actual popup text */
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 400px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            font-size: 24px;
            font-weight: normal;
            line-height: 1.4;

            /* Position the tooltip above the icon */
            position: absolute;
            z-index: 1;
            bottom: 125%; 
            left: 300%;
            margin-left: -100px; /* Centers the tooltip */

            /* Fade-in effect */
            opacity: 0;
            transition: opacity 0.3s;
        }

        /* Show the tooltip on hover */
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        /* Your existing circle icon style, adapted for the tooltip wrapper */
        .help-icon-circle {
            background-color: #e5987d; 
            color: white; 
            width: 40px; 
            height: 40px; 
            border-radius: 50%; 
            text-align: center; 
            line-height: 40px; 
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        # --- The Icon HTML ---
        st.markdown(
            '''
            <div class="tooltip">
                <div class="help-icon-circle">?</div>
                <span class="tooltiptext">
                This app helps parents and therapists create personalized Social Stories for children with autism.<br><br>
                Social Stories are short, supportive stories that explain everyday situations in a clear and calming way.
                helping children understand what to expect and prepare for upcoming changes in their daily lives<br><br>
                The stories are created based on well-known guidelines developed by Carol Gray and adapted to your child.<br><br>
                For your privacy and peace of mind, the system works offline and your data and images are not shared or sent outside the app.
                </span>
            </div>
            ''',
            unsafe_allow_html=True
        )

def show_Manage_Profiles():
    """Screen 2: Manage Profiles"""

    # --- TOP RIGHT LOGOUT BUTTON ---
    col_empty, col_logout = st.columns([10, 1.2])

    with col_logout:
        if st.button("Log Out", key="top_logout"):
            navigate_to('login')
            st.rerun()

    # --- Titles ---
    st.markdown("<h2>Manage Profiles</h2>", unsafe_allow_html=True)
    st.markdown("<h3>Select a Child Profile to Begin</h3>", unsafe_allow_html=True)

    st.write("")  # Small Spacer

    # --- Main Content Layout ---
    col_buttons, col_spacer, col_image = st.columns([1.2, 0.1, 0.8])

    with col_buttons:
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Get active profiles
        active_kids = st.session_state.data[st.session_state.data['Status'] == 'Active']

        # Create a button for each child
        for index, row in active_kids.iterrows():
            if st.button(row['Full Name'], key=f"select_{row['ID']}", width="stretch"):
                # Save the selected child's row to session state
                st.session_state.selected_child = row
                # Navigate to the dashboard
                navigate_to('child_profile')
                st.rerun()

        # Spacer
        st.markdown("<br>", unsafe_allow_html=True)

        # "Add New Profile" Button
        if st.button("Add a New Profile", width="stretch"):
            navigate_to('new_profile')
            st.rerun()

    with col_image:
        try:
            st.image("images/children_playing.png", width="stretch")
        except:
            st.info("Missing image: images/children_playing.png")


    # --- SUBTITLE WITH QUESTION MARK ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])
    with sub_col1:
        # --- Help Icon ---
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


def show_new_profile():
    """Screen 3: Add New Person Form"""

    # --- 0. Initialize variables ---
    name, age, gender, func_level, attention = None, None, None, None, None
    comp_level, read_level, interests, sensory, strengths = None, None, None, None, None
    general_details = None
    portrait_path = ""

    # --- CSS: Style Tweaks ---
    st.markdown("""
    <style>
    div[data-testid="column"] { padding: 0 !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0.3rem !important; }

    /* Hide the "Press Enter to apply" instructions */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Style the question mark */
    .tooltip-icon {
        font-size: 20px; color: #e5987d; cursor: help; margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Top Nav ---
    col_menu, col_title, col_back = st.columns([1, 10, 1])
    with col_back:
        if st.button("↩", key="back_btn_top"):
            navigate_to('Manage_Profiles')
            st.rerun()

    st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>Create a New Child Profile</h1>",
                unsafe_allow_html=True)

    with st.form("create_profile_form", clear_on_submit=False):

        # Main Columns
        c_image, c_input_left, c_input_right = st.columns([0.6, 2.2, 2.2], gap="small")

        # --- LEFT: Image ---
        with c_image:
            st.markdown("", unsafe_allow_html=True)
            try:
                st.image("images/boy.png", width="stretch")
            except:
                st.info("Missing 'images/boy.png'")

        # --- CENTER: Inputs ---
        with c_input_left:
            # 1. Name (MANDATORY *)
            name = st.text_input("Name", placeholder="Name *", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)

            # 2. Age (MANDATORY *)
            age = st.text_input("Age", placeholder="Age *", label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)

            # 3. Gender
            gender = st.selectbox("Gender", ["Gender", "Male", "Female", "Other"], label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)

            # 4. Functional Level
            sub_c1, sub_c2 = st.columns([9, 1])
            with sub_c1:
                func_level = st.selectbox("Functional Level", ["Functional Level", "High", "Moderate", "Low"],
                                          label_visibility="collapsed")
            with sub_c2:
                st.markdown("", help="How much help your child usually needs in daily activities.\n\n"
                                     "High – Mostly independent.\n\n"
                                     "Moderate – Needs some help or reminders.\n\n"
                                     "Low – Needs a lot of help and support")
            st.markdown("<br>", unsafe_allow_html=True)

            # 5. Attention Span
            sub_c1, sub_c2 = st.columns([9, 1])
            with sub_c1:
                attention = st.selectbox("Attention Span", ["Attention Span", "Short", "Medium", "Long"],
                                         label_visibility="collapsed")
            with sub_c2:
                st.markdown("", help="How long your child can usually stay focused on one activity.\n\n"
                                     "Short – A few minutes\n\n"
                                     "Medium – About 10–20 minutes\n\n"
                                     "Long – Can focus for a long time")

        # --- RIGHT: Inputs ---
        with c_input_right:
            # 1. Comprehension
            sub_c1, sub_c2 = st.columns([9, 1])
            with sub_c1:
                comp_level = st.selectbox("Comprehension Level",
                                          ["Comprehension Level", "Literal", "Abstract", "Mixed"],
                                          label_visibility="collapsed")
            with sub_c2:
                st.markdown("", help="How your child understands language and instructions.\n\n"
                                     "Literal – Understands things exactly as they are said\n\n"
                                     "Abstract – Can understand ideas like jokes, feelings, or “between the lines” meanings\n\n"
                                     "Mixed – Sometimes literal, sometimes abstract")
            st.markdown("<br>", unsafe_allow_html=True)

            # 2. Reading
            sub_c1, sub_c2 = st.columns([9, 1])
            with sub_c1:
                read_level = st.selectbox("Reading Level", ["Reading Level", "Pre-reader", "Early Reader", "Fluent"],
                                          label_visibility="collapsed")
            with sub_c2:
                st.markdown("", help="How comfortable your child is with reading.\n\n"
                                     "Pre-reader – Not reading yet\n\n"
                                     "Early reader – Can read simple words or short sentences\n\n"
                                     "Fluent reader – Reads and understands full sentences or short stories")
            st.markdown("<br>", unsafe_allow_html=True)

            # 3. Interests
            sub_c1, sub_c2 = st.columns([9, 1])
            with sub_c1:
                interests = st.text_input("Interests", placeholder="Interests", label_visibility="collapsed")
            with sub_c2:
                st.markdown("", help="Things your child enjoys or cares about.\n\n"
                                     "For example: favorite toys, characters, topics, or activities.")
            st.markdown("<br>", unsafe_allow_html=True)

            # 4. Sensory
            sub_c1, sub_c2 = st.columns([9, 1])
            with sub_c1:
                sensory = st.text_input("Sensory Needs", placeholder="Sensory Needs", label_visibility="collapsed")
            with sub_c2:
                st.markdown("", help="Any sensitivities, triggers or preferences, such as sensitivities to specific objects or animals, noise, light, textures.\n\n"
                                     "Or things that may feel calming for your child.\n\n")
            st.markdown("<br>", unsafe_allow_html=True)

            # 5. Strengths
            sub_c1, sub_c2 = st.columns([9, 1])
            with sub_c1:
                strengths = st.text_input("Strengths & Motivators", placeholder="Strengths & Motivators",
                                          label_visibility="collapsed")
            with sub_c2:
                st.markdown("", help="Things that help encourage your child and make learning easier.\n\n"
                                     "For example: praise, token, rewards, favorite activities, etc.")

        # --- Footer ---
        b_col1, b_col2, b_col3 = st.columns([1, 1, 1], gap="small")

        with b_col2:
            # General Details section
            st.markdown("", unsafe_allow_html=True)
            general_details = st.text_area(
                "General Details",
                placeholder="General Details: Add any extra information",
                label_visibility="collapsed",
                height=120
            )

            st.markdown("", unsafe_allow_html=True)

            # Upload profile Image
            u_col1, u_col2, u_col3 = st.columns([4.5, 0.5, 4.5])
            with u_col1:
                st.markdown("<b>Optional: Upload Child Portrait</b>", unsafe_allow_html=True)
            with u_col2:
                st.markdown("", help="All images are processed locally and the system works offline.\n\n"
                                     "Your images are not uploaded, shared, or stored outside this app.")


            uploaded_portrait = st.file_uploader(
                "Upload Portrait",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                label_visibility="collapsed"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Save button
            submitted = st.form_submit_button(
                "Save Profile",
                width="stretch"
            )

    # --- Cancel Button ---
    col_footer_left, col_footer_center, col_footer_right = st.columns([2, 1, 1])
    with col_footer_right:
        if st.button("Cancel", key="cancel_add"):
            navigate_to('Manage_Profiles')
            st.rerun()

    # --- Logic ---
    if submitted:
        # Check only Name and Age
        if name and age:
            new_id = str(1000 + len(st.session_state.data) + 1)

            if uploaded_portrait is not None:
                try:
                    img = Image.open(uploaded_portrait).convert("RGB")

                    # Center crop to square
                    w, h = img.size
                    side = min(w, h)
                    left = (w - side) // 2
                    top = (h - side) // 2
                    img = img.crop((left, top, left + side, top + side))

                    # Resize (future SDXL-friendly)
                    img = img.resize((1024, 1024))

                    out_path = PORTRAITS_DIR / f"{new_id}.jpg"
                    img.save(out_path, format="JPEG", quality=95)

                    portrait_path = str(out_path)

                except Exception:
                    st.error("Failed to process uploaded image.")
                    return

            # Creating the new entry
            new_entry = {
                'ID': new_id,
                'Full Name': name,
                'Age': age,
                'Gender': gender if gender != "Gender" else "Not Specified",
                'Status': 'Active',
                'Functional Level': func_level if func_level != "Functional Level" else "",
                'Attention Span': attention if attention != "Attention Span" else "",
                'Comprehension Level': comp_level if comp_level != "Comprehension Level" else "",
                'Reading Level': read_level if read_level != "Reading Level" else "",
                'Interests': interests,
                'Sensory Needs': sensory,
                'Strengths': strengths,
                'General Details': general_details,
                'Portrait Path': portrait_path,
            }

            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_entry])], ignore_index=True)

            st.success("Profile saved successfully!")
            time.sleep(2)
            navigate_to('Manage_Profiles')
            st.rerun()
        else:
            # Error specifically mentions Name and Age
            st.error("Please fill in the required fields: Name and Age.")


    # --- SUBTITLE WITH QUESTION MARK ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])

    with sub_col1:
        # --- Help Icon ---
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


def show_child_profile():
    """Screen: Child Dashboard"""

    # --- 1. Initialize Delete State ---
    if 'confirm_profile_delete' not in st.session_state:
        st.session_state.confirm_profile_delete = False

    # Check if a child is actually selected
    if 'selected_child' not in st.session_state:
        st.error("No child selected.")
        if st.button("Go Back"):
            navigate_to('Manage_Profiles')
            st.rerun()
        return

    child = st.session_state.selected_child

    # --- CSS Styling ---
    st.markdown("""
    <style>
    /* 1. Peach Buttons (Type = Secondary) - Used for Cancel and Menu */
    div.stButton > button[kind="secondary"] {
        background-color: #fce7db;
        color: #444;
        border: 1px solid #d4a796;
        border-radius: 15px;
        font-weight: bold;
        width: 100%;
        height: 45px; /* Fixed height for uniformity */
        margin-bottom: 10px;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #e6957a;
        border-color: #d48369;
    }

    /* 2. Red Delete Button (Type = Primary) - Used for Delete and Yes */
    div.stButton > button[kind="primary"] {
        background-color: #ff6b6b !important;
        color: white !important;
        /* Add transparent border so it matches the size of the secondary button's border */
        border: 1px solid transparent !important;
        font-weight: bold !important;
        width: 100% !important;
        height: 45px !important; /* Fixed height to match Secondary */
        border-radius: 15px !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #fa5252 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # --- Layout: 3 Columns ---
    c_left, c_center, c_right = st.columns([1, 2, 1], gap="medium")

    # --- LEFT COLUMN: Child Image ---
    with c_left:
        st.markdown("<br>" * 8, unsafe_allow_html=True)
        try:
            st.image("images/reading_child.png", width="stretch")
        except:
            st.info("Missing 'reading_child.png'")

    # --- CENTER COLUMN: Title AND Buttons ---
    with c_center:
        # 1. Titles
        st.markdown(f"<h1 style='text-align: center; margin-bottom: 0;'>-{child['Full Name']}-</h1>",
                    unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; font-size: 2em !important; margin-top: 0;'>Welcome Back!</h2>",
                    unsafe_allow_html=True)
        st.markdown(
            "<h3 style='text-align: center; font-style: italic; margin-bottom: 30px;'>What Would You Like To Do Today?</h3>",
            unsafe_allow_html=True)

        # 2. Buttons

        # Button A: Create New Story -> Goes to Story Display
        if st.button("Create New Story", type="secondary", width="stretch"):
            navigate_to('create_new_story')
            st.rerun()

        # Button B: Saved Stories -> Goes to Saved Story View
        if st.button("Saved Stories", type="secondary", width="stretch"):
            navigate_to('saved_stories_list')
            st.rerun()

        # Button C: Update Profile
        if st.button("Update Profile Details", type="secondary", width="stretch"):
            st.toast("Edit profile clicked")

    # --- RIGHT COLUMN: Back, Books, Delete ---
    with c_right:
        # Top Row inside Right Column for the Back Button
        r_col1, r_col2 = st.columns([1, 1])
        with r_col2:
            if st.button("↩", key="back_to_list", type="secondary"):
                # Reset delete state if user leaves
                st.session_state.confirm_profile_delete = False
                navigate_to('Manage_Profiles')
                st.rerun()

        # Books stack (Top aligned)
        try:
            st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
            st.image("images/books_stack.png", width=120)
            st.markdown("</div>", unsafe_allow_html=True)
        except:
            st.info("Missing 'books_stack.png'")

        # Push Delete button to bottom
        st.markdown("<br>" * 12, unsafe_allow_html=True)

        # --- DELETE BUTTON LOGIC ---
        d_col1, d_col2 = st.columns([0.2, 2])
        with d_col2:

            # State A: Normal State (Show Delete Button)
            if not st.session_state.confirm_profile_delete:
                # This uses the RED style (primary)
                if st.button("Delete Profile", key="delete_profile_init", type="primary", width="stretch"):
                    st.session_state.confirm_profile_delete = True
                    st.rerun()

            # State B: Confirmation State
            else:
                st.markdown(
                    "<div style='text-align: center; color: #d9534f; font-weight: bold; margin-bottom: 5px; font-size: 0.8em;'>Are you sure you want<br>to delete this profile?</div>",
                    unsafe_allow_html=True
                )

                col_yes, col_cancel = st.columns(2)

                # --- Create a placeholder BELOW the buttons for the message ---
                msg_box = st.empty()

                with col_yes:
                    # Yes = Red (Primary)
                    if st.button("Yes", key="del_prof_yes", type="primary", width="stretch"):
                        # 1. Remove child from data
                        st.session_state.data = st.session_state.data[st.session_state.data['ID'] != child['ID']]
                        # 2. Clear selected child
                        del st.session_state.selected_child
                        # 3. Reset flag
                        st.session_state.confirm_profile_delete = False

                        # 4. Display success message in the wide placeholder below
                        msg_box.success("Profile Deleted")

                        time.sleep(2)
                        navigate_to('Manage_Profiles')
                        st.rerun()

                with col_cancel:
                    # Cancel = Standard Peach => Go back
                    if st.button("Cancel", key="del_prof_cancel", width="stretch"):
                        st.session_state.confirm_profile_delete = False
                        st.rerun()

    # --- SUBTITLE WITH HELP ICON ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])

    with sub_col1:
        # Help Icon styling (Non-functional, visual only)
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


def show_create_new_story():
    """Slide 6: Create New Story – Template Selection"""

    # --- TOP BAR ---
    top_left, top_center, top_right = st.columns([1, 6, 1])

    with top_left:
        st.image("images/menu_bar.png", width=40)


    with top_right:
        if st.button("↩", key="back_to_list", type="secondary"):
            # Reset delete state if user leaves
            st.session_state.confirm_profile_delete = False
            navigate_to('child_profile')
            st.rerun()


    st.markdown("<br>", unsafe_allow_html=True)

    # --- TITLE ---
    st.markdown(
        "<h1 style='text-align:center;'>Create a New Story</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='text-align:center;'>Choose a template to get started quickly, or create a custom story from scratch</h2>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<h3 style='text-align:center;'>Ready-to-use Templates</h3>",
        unsafe_allow_html=True
    )


    # --- STORY TEMPLATE CARDS ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("images/New_School.png", width="stretch")
        if st.button("Starting a New School", width="stretch"):
            st.session_state.selected_template = "new_school"
            navigate_to("new_school_story")
            st.rerun()

    with col2:
        st.image("images/New_Pet.png", width="stretch")
        if st.button("Getting a New Pet", width="stretch"):
            st.session_state.selected_template = "new_pet"
            navigate_to("new_pet_story")
            st.rerun()

    with col3:
        st.image("images/New_Baby.png", width="stretch")
        if st.button("A New Baby Sibling", width="stretch"):
            st.session_state.selected_template = "new_baby"
            navigate_to("new_baby_story")
            st.rerun()

    # Add space between buttons
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- CREATE CUSTOM STORY ---
    custom_col1, custom_col2, custom_col3 = st.columns([2, 3, 2])

    with custom_col2:
        if st.button("Create a Custom Story", width="stretch"):
            st.session_state.selected_template = "custom"
            navigate_to("custom_story_details")
            st.rerun()

    # --- SUBTITLE WITH QUESTION MARK ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])

    with sub_col1:
        # --- Help Icon ---
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


def show_new_school_story():
    """Slide 7: Customize Your Story – Starting New School"""

    # --- TOP BAR ---
    top_left, top_center, top_right = st.columns([1, 6, 1])

    with top_left:
        st.image("images/menu_bar.png", width=40)

    with top_right:
        if st.button("↩", key="back_to_list", type="secondary"):
            # Reset delete state if user leaves
            st.session_state.confirm_profile_delete = False
            navigate_to('create_new_story')
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TITLE ---
    st.markdown(
        "<h1 style='text-align:center;'>Customize Your Story</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='text-align:center;'>Starting a New School</h2>",
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    left_col, right_col = st.columns([1.2, 1.8])

    # LEFT — IMAGE
    with left_col:
        st.image("images/New_School.png", width="stretch")

    # RIGHT — INPUTS
    with right_col:

        # Type of School (WITH HELP)
        sub_c1, sub_c2 = st.columns([9, 1])

        with sub_c1:
            school_type = st.text_input(
                "Type of School",
                placeholder="Type of School",
                key="school_type",
                label_visibility="collapsed"
            )

        with sub_c2:
            st.markdown(
                "",
                help="For example: elementary school, kindergarden, special education school."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # School name
        school_name = st.text_input(
            "School’s Name",
            placeholder="School’s Name",
            key="school_name",
            label_visibility="collapsed"
        )

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("### Anything else you’d like to add?")

        additional_notes = st.text_area(
            "school notes",
            height=260,   # box size control
            key="school_additional_notes",
            placeholder="e.g. New teachers, new classroom, worried about making friends...",
            label_visibility="collapsed"
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- CTA BUTTON ---
    btn_col1, btn_col2, btn_col3 = st.columns([2, 3, 2])
    with btn_col2:
        if st.button("Generate A Story", width="stretch"):

            if not st.session_state.get("school_type"):
                st.warning("Please fill in the type of school.")
            else:
                child_profile = get_child_profile_str()
                school_context = build_new_school_context()

                # --- CRITICAL: Save for regeneration ---
                st.session_state.initial_story_context = school_context

                # --- Reset Q&A state (template story) ---
                st.session_state.llm_questions = []
                st.session_state.llm_answers = []
                st.session_state.question_index = 0
                st.session_state.is_thinking = False

                final_full_context = f"{child_profile}\n\nSITUATION DETAILS:\n{school_context}"
                final_prompt = build_final_story_prompt(final_full_context)

                with st.spinner("Writing your personalized story..."):
                    st.session_state.story_display = ask_ollama(final_prompt)
                    st.session_state.story_title = "Starting a New School"

                st.session_state.pop("story_image", None)
                navigate_to("story_display")
                st.rerun()

    # --- QUESTION MARK ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])

    with sub_col1:
        # --- Help Icon ---
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


def show_new_pet_story():
    """Slide 8: Customize Your Story – Getting New Pet (FIXED)"""

    # --- TOP BAR ---
    top_left, top_center, top_right = st.columns([1, 6, 1])

    with top_left:
        st.image("images/menu_bar.png", width=40)

    with top_right:
        if st.button("↩", key="back_to_list", type="secondary"):
            st.session_state.confirm_profile_delete = False
            navigate_to('create_new_story')
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TITLE ---
    st.markdown("<h1 style='text-align:center;'>Customize Your Story</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Getting a New Pet</h2>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    left_col, right_col = st.columns([1.2, 1.8])

    with left_col:
        try:
            st.image("images/New_Pet.png", width="stretch")
        except:
            st.info("Img placeholder")

    with right_col:
        pet_type_input = st.text_input("Type of Pet", placeholder="Type of Pet (e.g. Dog, Cat)", key="pet_type", label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        pet_name_input = st.text_input("Pet’s Name", placeholder="Pet’s Name", key="pet_name",label_visibility="collapsed")
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown("### Anything else you’d like to add?")
        additional_notes = st.text_area("additional notes", height=260, key="pet_additional_notes",
                                        placeholder="e.g. The dog is very big but friendly...", label_visibility="collapsed")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- CTA BUTTON ---
    btn_col1, btn_col2, btn_col3 = st.columns([2, 3, 2])
    with btn_col2:
        if st.button("Generate A Story", width="stretch"):

            if not st.session_state.get("pet_type") or not st.session_state.get("pet_name"):
                st.warning("Please fill in the Pet Type and Name.")
            else:
                child_profile = get_child_profile_str()
                pet_context = build_new_pet_context()

                # --- Save context so Regenerate can find it ---
                st.session_state.initial_story_context = pet_context
                # ---------------------------------------------------------------

                # Clear old Q&A history
                st.session_state.llm_questions = []
                st.session_state.llm_answers = []

                final_full_context = f"{child_profile}\n\nSITUATION DETAILS:\n{pet_context}"
                final_prompt = build_final_story_prompt(final_full_context)

                with st.spinner("Writing your personalized story..."):
                    st.session_state.story_display = ask_ollama(final_prompt)
                    # Save the title for display immediately
                    st.session_state.story_title = f"Getting a New {st.session_state.get('pet_type')}"

                st.session_state.pop("story_image", None)
                navigate_to("story_display")
                st.rerun()

    # --- QUESTION MARK ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])
    with sub_col1:
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


def show_new_baby_story():
    """Slide 9: Customize Your Story – Getting New Sibling"""

    # --- TOP BAR ---
    top_left, top_center, top_right = st.columns([1, 6, 1])

    with top_left:
        st.image("images/menu_bar.png", width=40)

    with top_right:
        if st.button("↩", key="back_to_list", type="secondary"):
            # Reset delete state if user leaves
            st.session_state.confirm_profile_delete = False
            navigate_to('create_new_story')
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TITLE ---
    st.markdown(
        "<h1 style='text-align:center;'>Customize Your Story</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='text-align:center;'>Having a New Baby Sibling</h2>",
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    left_col, right_col = st.columns([1.2, 1.8])

    # LEFT — IMAGE
    with left_col:
        st.image("images/New_Baby.png", width="stretch")

    # RIGHT — INPUTS
    with right_col:
        # Baby's Gender (switch-style)
        gender = st.selectbox(
            "Baby's Gender",
            ["Baby's Gender", "Male", "Female", "Unknown"],
            key="baby_gender",
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### Anything else you’d like to add?")

        additional_notes = st.text_area(
            "baby notes",
            height=260,   # box size control
            key="baby_additional_notes",
            placeholder="e.g. The baby will need a lot of attention at first...",
            label_visibility="collapsed"
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- CTA BUTTON ---
    btn_col1, btn_col2, btn_col3 = st.columns([2, 3, 2])
    with btn_col2:
        if st.button("Generate A Story", width="stretch"):

            if not st.session_state.get("baby_gender") or st.session_state.get("baby_gender") == "Baby's Gender":
                st.warning("Please select the baby's gender.")
            else:
                child_profile = get_child_profile_str()
                sibling_context = build_new_sibling_context()

                # --- CRITICAL: save context for regeneration ---
                st.session_state.initial_story_context = sibling_context

                # --- Reset Q&A state (template story) ---
                st.session_state.llm_questions = []
                st.session_state.llm_answers = []
                st.session_state.question_index = 0
                st.session_state.is_thinking = False

                final_full_context = f"{child_profile}\n\nSITUATION DETAILS:\n{sibling_context}"
                final_prompt = build_final_story_prompt(final_full_context)

                with st.spinner("Writing your personalized story..."):
                    st.session_state.story_display = ask_ollama(final_prompt)
                    st.session_state.story_title = "Having a New Baby Sibling"

                st.session_state.pop("story_image", None)
                navigate_to("story_display")
                st.rerun()

    # --- QUESTION MARK ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])

    with sub_col1:
        # --- Help Icon ---
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


def show_custom_story_details():
    """Create a Custom Story – Situation Details"""

    # --- TOP BAR ---
    top_left, top_center, top_right = st.columns([1, 6, 1])
    with top_left:
        st.image("images/menu_bar.png", width=40)
    with top_right:
        if st.button("↩", key="back_to_list", type="secondary"):
            # Reset delete state if user leaves
            st.session_state.confirm_profile_delete = False
            navigate_to('create_new_story')
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TITLE ---
    st.markdown("<h1 style='text-align:center;'>Create a Custom Story</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center; font-style: italic;'>"
        "Tell us about the situation, so we can create a personalized Social Story"
        "</h3>",
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- CENTERED MAIN CONTENT ---
    outer_l, content, outer_r = st.columns([2, 6, 2])

    with content:
        # --- ROW 1: Situation ---
        row1_l, row1_r = st.columns([2.2, 4.8])
        with row1_l:
            st.markdown(
                "<p style='font-size:16px; font-weight:600; margin:0;'>"
                "What situation do you like the story to address?"
                "</p>",
                unsafe_allow_html=True
            )
        with row1_r:
            situation = st.text_input(
                "situation title",
                placeholder="e.g: going to the dentist, difficulty sharing toys...",
                label_visibility="collapsed"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- ROW 2: Involved ---
        row2_l, row2_r = st.columns([2.2, 4.8])
        with row2_l:
            st.markdown(
                "<p style='font-size:16px; font-weight:600; margin:0;'>"
                "Who is involved, and what are their roles?"
                "</p>",
                unsafe_allow_html=True
            )
        with row2_r:
            involved = st.text_area(
                "situation participants",
                placeholder=(
                    "e.g:\n"
                    "- Adam (the child): gets nervous when it’s loud.\n"
                    "- Mom: supports Adam before school.\n"
                    "- Teacher (Ms. Sarah): helps guide him in class.\n"
                    "- Classmate (Liam): plays with Adam during recess."
                ),
                height=260,
                label_visibility="collapsed"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # --- ROW 3: Extra Info ---
        row3_l, row3_r = st.columns([2.2, 4.8])
        with row3_l:
            st.markdown(
                "<p style='font-size:16px; font-weight:600; margin:0;'>"
                "Anything else you'd like us to know?"
                "</p>",
                unsafe_allow_html=True
            )
        with row3_r:
            extra_info  = st.text_area(
                "situation details",
                placeholder=(
                    "e.g: The situation involves taking turns when playing a board game with a friend."
                ),
                height=100,
                label_visibility="collapsed"
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- CTA BUTTON (Next) ---
    btn_l, btn_c, btn_r = st.columns([4, 3, 4])
    with btn_c:
        if st.button("Next", width="stretch"):
            # --- Save the inputs to session state ---
            # Combine them into a single context string for the LLM
            user_input_summary = (
                f"Situation: {situation}\n"
                f"Who is involved: {involved}\n"
                f"Extra Info: {extra_info}"
            )
            st.session_state.initial_story_context = user_input_summary

            # --- Reset LLM State for a fresh start ---
            # Ensures the next screen starts from Question 0
            st.session_state.llm_questions = []
            st.session_state.llm_answers = []
            st.session_state.question_index = 0
            st.session_state.is_thinking = False

            # --- Navigate and Rerun ---
            navigate_to("additional_details_1")
            st.rerun()

    # --- BOTTOM BAR ---
    bottom_l, bottom_c, bottom_r = st.columns([1, 6, 1])

    with bottom_l:
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )

    with bottom_r:
        st.markdown("", unsafe_allow_html=True)
        try:
            st.image("images/books_stack2.png", width=200)
        except:
            pass


def show_additional_details_1():
    """Additional Details – Guided Questions"""

    # --- TOP BAR ---
    top_left, top_center, top_right = st.columns([1, 6, 1])
    with top_left:
        st.image("images/menu_bar.png", width=40)
    with top_right:
        if st.button("↩", key="back_to_list", type="secondary"):
            st.session_state.confirm_profile_delete = False
            navigate_to('custom_story_details')
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- TITLE ---
    st.markdown("<h1 style='text-align:center;'>Additional Details</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align:center; font-style: italic;'>"
        "It will help personalize the story for your child"
        "</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align:center; font-style: italic;'>"
        "We may ask up to 3 short questions. You can skip any question at any time."
        "</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- STOP CONDITION ---
    if st.session_state.question_index >= 2:
        st.session_state.is_thinking = False
        navigate_to("additional_details_2")
        st.rerun()
        return

    # --- PROGRESS BAR ---
    progress_bar = st.empty()
    if st.session_state.is_thinking:
        current_progress = (st.session_state.question_index + 1) / 3
    else:
        current_progress = st.session_state.question_index / 3
    progress_bar.progress(min(current_progress, 1.0))

    if st.session_state.is_thinking:
        st.markdown("<p style='text-align:center; font-size:16px; color:#888;'>Thinking…</p>", unsafe_allow_html=True)

    # --- BOOTSTRAP FIRST QUESTION ---
    if not st.session_state.llm_questions:
        # --- SPINNER SHOWS WHILE LOADING THE MODEL/FIRST QUESTION ---
        with st.spinner("Analyzing your request to generate the best questions..."):
            # 1. Get the Child Profile String
            child_profile = get_child_profile_str()

            # 2. Get the Custom Situation context (saved in previous screen)
            situation_details = st.session_state.get("initial_story_context", "General situation")

            # 3. Combine them into one big context for the LLM
            full_context = f"{child_profile}\n\nSITUATION DETAILS:\n{situation_details}"

            # 4. Generate the prompt
            first_prompt = build_question_prompt(full_context)

            first_q = ask_ollama(first_prompt)
            st.session_state.llm_questions.append(first_q)
            st.session_state.question_index = 0
            st.rerun()

    if not st.session_state.is_thinking:

        # --- CURRENT QUESTION ---
        current_question = st.session_state.llm_questions[
            st.session_state.question_index
        ]

        outer_l, content, outer_r = st.columns([2, 6, 2])
        with content:
            st.markdown(
                f""" <div style=' background-color:#f7cdb3; border-radius:18px;
                 padding:18px; border:1px solid #e5987d; text-align:center;
                  font-size:18px; font-weight:600; color:#666; '> {current_question} 
                  </div> """,
                unsafe_allow_html=True
            )

        st.markdown("<br><br>", unsafe_allow_html=True)

        # Controlled input
        answer_key = f"answer_{st.session_state.question_index}"
        answer = st.text_area(
            "Answer",
            placeholder="Answer",
            key=answer_key,
            height=140,
            label_visibility="collapsed"
        )

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- ACTION BUTTONS ---
        btn_l, btn_c, btn_r = st.columns([2, 3, 2])
        with btn_c:
            next_clicked = st.button("Next Question", width="stretch")
            if next_clicked:
                st.session_state.pending_answer = answer
                st.session_state.is_thinking = True
                st.rerun()

        with btn_r:
            if st.button("Skip", width="stretch"):
                st.session_state.pending_answer = ""
                st.session_state.is_thinking = True
                st.rerun()

        # --- BOTTOM ACTION ---
        bottom_l, bottom_c, bottom_r = st.columns([1, 6, 2])
        with bottom_l:
            st.markdown(
                '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
                unsafe_allow_html=True
            )
        with bottom_r:
            if st.button("Skip All & Generate A Story", width="stretch"):
                progress_bar.progress(1.0)

                with st.spinner("Generating your story..."):
                    # Fill blanks
                    while len(st.session_state.llm_answers) < len(st.session_state.llm_questions):
                        st.session_state.llm_answers.append("")

                    # --- final generation ---
                    st.session_state.story_display = generate_story_from_session()

                    st.session_state.question_index = 3
                    st.session_state.is_thinking = False
                    navigate_to("story_display")
                    st.rerun()

    # --- RUN LLM AFTER UI (Next Question) ---
    if st.session_state.is_thinking:
        # Spinner for subsequent questions is handled here
        with st.spinner("Thinking..."):
            st.session_state.llm_answers.append(st.session_state.pending_answer)

            # Context for the NEXT question also needs the profile
            child_profile = get_child_profile_str()
            situation_details = st.session_state.get("initial_story_context", "")

            qa_context = ""
            for q, a in zip(st.session_state.llm_questions, st.session_state.llm_answers):
                qa_context += f"Q: {q}\nA: {a}\n"

            full_context_for_next_q = f"{child_profile}\n\nSITUATION:\n{situation_details}\n\nHISTORY:\n{qa_context}"

            next_prompt = build_question_prompt(full_context_for_next_q)
            next_q = ask_ollama(next_prompt)
            st.session_state.llm_questions.append(next_q)

            st.session_state.question_index += 1
            st.session_state.is_thinking = False
            st.rerun()


def show_additional_details_2():
    """Additional Details – Final Question (Q3)"""

    # --- GUARD: only allow entry for Q3 ---
    if st.session_state.question_index < 2:
        navigate_to("additional_details_1")
        st.rerun()
        return

    # --- STOP: after Q3, go to final story ---
    if st.session_state.question_index >= 3:
        navigate_to("story_display")
        st.rerun()

    # --- TITLE ---
    st.markdown(
        "<h1 style='text-align:center;'>Final Detail</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align:center; font-style: italic;'>"
        "Just one last question before we create the story:"
        "</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- PROGRESS BAR ---
    progress_bar = st.empty()

    if st.session_state.is_thinking:
        # Generating final story -> 100%
        progress_bar.progress(1.0)
        st.markdown(
            "<p style='text-align:center; font-size:16px; color:#888;'>Generating Story...</p>",
            unsafe_allow_html=True
        )
    else:
        # Q3 (Index 2) -> approx 66%
        progress_bar.progress(0.66)

    # --- INTERACTIVE UI ---
    if not st.session_state.is_thinking:

        current_question = st.session_state.llm_questions[
            st.session_state.question_index
        ]

        outer_l, content, outer_r = st.columns([2, 6, 2])
        with content:
            st.markdown(
                f"""
                <div style='
                    background-color:#f7cdb3;
                    border-radius:18px;
                    padding:18px;
                    border:1px solid #e5987d;
                    text-align:center;
                    font-size:18px;
                    font-weight:600;
                    color:#666;
                '>
                    {current_question}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br><br>", unsafe_allow_html=True)

            answer_key = f"answer_{st.session_state.question_index}"
            answer = st.text_area(
                "Answer",
                placeholder="Answer (optional)",
                key=answer_key,
                height=140,
                label_visibility="collapsed"
            )

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- ACTION BUTTONS ---
        btn_l, btn_c, btn_r = st.columns([2, 3, 2])

        with btn_c:
            if st.button("Generate Story", width="stretch"):
                # Force bar to 100%
                progress_bar.progress(1.0)

                st.session_state.pending_answer = answer
                st.session_state.is_thinking = True
                st.rerun()

    # --- RUN LLM AFTER UI (Generate Final Story) ---
    if st.session_state.is_thinking:
        # 1. Save the final answer
        st.session_state.llm_answers.append(
            st.session_state.pending_answer
        )

        # 2. Add Spinner for UX
        with st.spinner("Writing your personalized story..."):

            # --- CRITICAL UPDATE START ---
            st.session_state.story_display = generate_story_from_session()
            st.session_state.pop("story_image", None)

            # --- CRITICAL UPDATE END ---

        # 3. Mark flow complete
        st.session_state.question_index += 1
        st.session_state.is_thinking = False

        # 4. Navigate
        navigate_to("story_display")
        st.rerun()


def show_story_display():
    """Screen 7: Display Generated Story (With Title Extraction)"""

    # --- Header: Back Button ---
    col_l, col_r = st.columns([10, 1])
    with col_r:
        if st.button("↩", key="back_from_story"):
            navigate_to("child_profile")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 1. EXTRACT TITLE AND CONTENT ---
    raw_text = st.session_state.get("story_display", "")

    # Defaults
    final_title = "Your Story"
    final_content = "Your story is being prepared..."

    # --- Auto-generate illustration ONLY if portrait exists ---
    child = st.session_state.selected_child
    portrait_path = child.get("Portrait Path", "")
    has_portrait = bool(portrait_path and Path(portrait_path).exists())

    if has_portrait and "story_image" not in st.session_state:
        with st.spinner("Generating Image"):
            story_text = st.session_state.get("story_display", "")
            child_profile = get_child_profile_str()

            image_prompt = ask_ollama(
                build_image_prompt(
                    story_text=story_text,
                    child_profile=child_profile,
                    has_portrait=True
                )
            )

            st.session_state.story_image = generate_story_image(
                prompt=image_prompt,
                portrait_path=portrait_path
            )

    if raw_text:
        # Try to split the first line from the rest
        # Split by the first newline character ('\n')
        parts = raw_text.split('\n', 1)

        if len(parts) >= 2:
            # Found a title and content
            # Clean up title (remove **, ##, or "Title:")
            extracted_title = parts[0].strip().replace("**", "").replace("Title:", "").strip()
            extracted_content = parts[1].strip()

            final_title = extracted_title
            final_content = extracted_content
        else:
            # Fallback if AI didn't add a newline
            final_content = raw_text

    # --- 2. DISPLAY TITLE (H1) ---
    st.markdown(
        f"<h1 style='text-align:center;'>{final_title}</h1>",
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- 3. DISPLAY CONTENT (Box) ---
    with st.container(border=True):
        st.markdown(
            f"""
            <div style='
                font-size: 1.1em;
                line-height: 1.6;
                padding: 20px;
                background-color: white;
                border-radius: 10px;
            '>
                {final_content}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if "story_image" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_c, col_r = st.columns([2, 2, 2])
        with col_c:
            st.image(
                st.session_state.story_image,
                width="stretch"
            )

    # --- Feedback Section ---
    st.markdown(
        "<h3 style='text-align: center;'>Did you like the story?</h3>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Layout: margins | like | gap | dislike | margins
    c_margin_l, c_like, c_gap, c_dislike, c_margin_r = st.columns(
        [1.5, 1, 0.2, 1, 1.5]
    )

    # --- LIKE ---
    with c_like:
        sub_l, sub_img, sub_r = st.columns([1, 2, 1])
        with sub_img:
            try:
                st.image("images/like_icon.png", width=80)
            except Exception:
                st.write("👍")

        if st.button("Like!", key="btn_like_story", width="stretch"):
            navigate_to("like_story")
            st.rerun()

    # --- DISLIKE ---
    with c_dislike:
        sub_l, sub_img, sub_r = st.columns([1, 2, 1])
        with sub_img:
            try:
                st.image("images/dislike_icon.png", width=80)
            except Exception:
                st.write("👎")

        if st.button("Dislike", key="btn_dislike_story", width="stretch"):
            navigate_to("dislike_story")
            st.rerun()

    # --- Help Icon ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])

    with sub_col1:
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


# TODO: Menu, add. function?, prompt as txt. file?,
# TODO: update prompts, sensory needs
def show_like_story():
    """Screen 5: Story Finished (Save/Discard)"""

    # --- 1. Initialize Confirmation State ---
    if 'confirm_delete_finished' not in st.session_state:
        st.session_state.confirm_delete_finished = False

    # --- CSS for the Primary button (Red) ---
    st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #fa5252 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header: Menu & Back Button ---
    col_l, col_r = st.columns([10, 1])
    with col_r:
        if st.button("↩", key="back_from_finish"):
            st.session_state.confirm_delete_finished = False
            navigate_to('story_display')
            st.rerun()

    st.write("")  # Spacer

    # --- Centered Text ---
    st.markdown("<h2 style='text-align: center;'>We are glad that you enjoyed our story!</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Would you like to save the story?</h3>", unsafe_allow_html=True)

    st.write("")  # Spacer

    # --- Buttons Layout (WIDE SPACING) ---
    c_margin_l, c_save, c_gap, c_discard, c_margin_r = st.columns([1, 2, 3, 2, 1])

    # --- Save Button (UPDATED LOGIC) ---
    with c_save:
        if st.button("Save", key="btn_save_finished", width="stretch"):

            # --- 1. Extract Data ---
            raw_text = st.session_state.get("story_display", "")

            # Defaults
            title = "Untitled Story"
            content = raw_text

            # Try to split Title (Line 1) from Content
            if raw_text:
                parts = raw_text.split('\n', 1)
                if len(parts) >= 2:
                    title = parts[0].strip().replace("**", "").replace("Title:", "").strip()
                    content = parts[1].strip()

            # --- 2. SAVE IMAGE TO DISK ---
            saved_image_path = None
            if "story_image" in st.session_state and st.session_state.story_image:
                try:
                    # Create a unique filename using timestamp
                    timestamp = int(time.time())
                    filename = f"story_{timestamp}_{st.session_state.selected_child['ID']}.png"
                    save_path = STORY_IMAGES_DIR / filename

                    # Save the PIL image to the uploads folder
                    st.session_state.story_image.save(save_path)
                    saved_image_path = str(save_path)
                except Exception as e:
                    print(f"Error saving image: {e}")

            # --- 3. Create Story Object ---
            new_story = {
                "id": int(time.time()),  # Unique ID based on time
                "child_id": st.session_state.selected_child['ID'],
                "title": title,
                "content": content,
                "date": time.strftime("%Y-%m-%d"),
                "image_path": saved_image_path
            }

            # --- 4. Save to Database ---
            if 'saved_stories' not in st.session_state:
                st.session_state.saved_stories = []

            st.session_state.saved_stories.append(new_story)

            # --- 4. Success & Navigate ---
            st.success("Your story has been saved, going back to home page!")
            time.sleep(2)
            st.session_state.confirm_delete_finished = False
            navigate_to('child_profile')
            st.rerun()

    # --- Discard Button (WITH CONFIRMATION LOGIC) ---
    with c_discard:
        if not st.session_state.confirm_delete_finished:
            if st.button("Discard", key="btn_discard_finished", width="stretch"):
                st.session_state.confirm_delete_finished = True
                st.rerun()
        else:
            st.markdown(
                "<div style='text-align: center; color: #d9534f; font-weight: bold; margin-bottom: 5px; font-size: 0.9em;'>You chose to delete the story.<br>Are you sure?</div>",
                unsafe_allow_html=True
            )
            col_yes, col_cancel = st.columns(2)
            with col_yes:
                if st.button("Yes", key="confirm_yes_finished", type="primary", width="stretch"):
                    st.success("Deleted!")
                    time.sleep(2)
                    st.session_state.confirm_delete_finished = False
                    navigate_to('child_profile')
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key="confirm_cancel_finished", width="stretch"):
                    st.session_state.confirm_delete_finished = False
                    st.rerun()

    # --- Bottom Image ---
    st.markdown("<br>", unsafe_allow_html=True)
    col_spacer_l, col_img, col_spacer_r = st.columns([1, 1, 1])
    with col_img:
        try:
            st.image("images/boy2.png", width="stretch")
        except:
            st.info("Missing image placeholder")

    # --- Help Icon ---
    st.markdown(
        '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
        unsafe_allow_html=True
    )


def show_dislike_story():
    """Screen 6: Feedback & Regenerate Story (Clean Version)"""

    # --- 1. Initialize Confirmation State ---
    if 'confirm_delete' not in st.session_state:
        st.session_state.confirm_delete = False

    # --- CSS for Primary button ---
    st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #fa5252 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    col_l, col_r = st.columns([10, 1])
    with col_r:
        if st.button("↩", key="back_from_feedback"):
            st.session_state.confirm_delete = False
            navigate_to('story_display')
            st.rerun()

    st.write("")

    # --- Titles ---
    st.markdown("<h2 style='text-align: center;'>We're sorry the story wasn't quite right</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size: 1.2em;'>How can we make it better?</h3>",
                unsafe_allow_html=True)

    st.write("")

    # --- Feedback Input ---
    c_spacer_l, c_input, c_spacer_r = st.columns([1, 6, 1])
    with c_input:
        feedback = st.text_area("Feedback", placeholder="Describe what needs to be improved...", height=100,
                                label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size: 1.2em;'>What would you like to do next?</h3>",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- 1. REGENERATE BUTTON (Refactored) ---
    c_r1, c_regen, c_r2 = st.columns([1, 1.5, 1])

    with c_regen:
        if st.button("Regenerate Story", key="btn_regen", width="stretch"):

            if not feedback:
                st.warning("Please enter some feedback first.")
            else:
                with st.spinner("Regenerating story based on your feedback..."):

                    # --- CALL THE NEW HELPER FUNCTION ---
                    final_prompt = build_regeneration_full_prompt(feedback)

                    # --- GENERATE ---
                    new_story = ask_ollama(final_prompt)

                    # --- UPDATE & NAVIGATE ---
                    st.session_state.story_display = new_story
                    st.toast("Story updated!")
                    time.sleep(2)
                    st.session_state.pop("story_image", None)
                    navigate_to('story_display')
                    st.rerun()

        st.markdown(
            "<p style='text-align: center; font-size: 0.8em; color: #666; margin-top: 5px;'>Use your feedback to<br>improve this story</p>",
            unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- 2. SAVE & DISCARD BUTTONS (Standard Logic) ---
    c_margin_l, c_save, c_gap, c_discard, c_margin_r = st.columns([1, 2, 3, 2, 1])

    # --- Save Button ---
    with c_save:
        if st.button("Save Story", key="btn_save_feedback", width="stretch"):

            # Extract
            raw_text = st.session_state.get("story_display", "")
            title = "Untitled Story"
            content = raw_text
            if raw_text:
                parts = raw_text.split('\n', 1)
                if len(parts) >= 2:
                    title = parts[0].strip().replace("**", "").replace("Title:", "").strip()
                    content = parts[1].strip()

            # --- SAVE IMAGE (New Code) ---
            saved_image_path = None
            if "story_image" in st.session_state and st.session_state.story_image:
                try:
                    timestamp = int(time.time())
                    filename = f"story_{timestamp}_{st.session_state.selected_child['ID']}.png"
                    save_path = STORY_IMAGES_DIR / filename
                    st.session_state.story_image.save(save_path)
                    saved_image_path = str(save_path)
                except Exception as e:
                    print(f"Error saving image: {e}")

            # Create Object
            new_story = {
                "id": int(time.time()),
                "child_id": st.session_state.selected_child['ID'],
                "title": title,
                "content": content,
                "date": time.strftime("%Y-%m-%d"),
                "image_path": saved_image_path
            }

            # Save
            if 'saved_stories' not in st.session_state:
                st.session_state.saved_stories = []
            st.session_state.saved_stories.append(new_story)

            st.success("Saved to Library!")
            time.sleep(2)
            st.session_state.confirm_delete = False
            navigate_to('child_profile')
            st.rerun()

    # --- Discard Button ---
    with c_discard:
        if not st.session_state.confirm_delete:
            if st.button("Discard Story", key="btn_discard_feedback", width="stretch"):
                st.session_state.confirm_delete = True
                st.rerun()
        else:
            st.markdown(
                "<div style='text-align: center; color: #d9534f; font-weight: bold; margin-bottom: 5px; font-size: 0.9em;'>You chose to delete the story.<br>Are you sure?</div>",
                unsafe_allow_html=True
            )
            col_yes, col_cancel = st.columns(2)
            with col_yes:
                if st.button("Yes", key="confirm_yes", type="primary", width="stretch"):
                    st.success("Deleted!")
                    time.sleep(2)
                    st.session_state.confirm_delete = False
                    navigate_to('child_profile')
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key="confirm_cancel", width="stretch"):
                    st.session_state.confirm_delete = False
                    st.rerun()

    # --- Help Icon ---
    st.markdown(
        '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
        unsafe_allow_html=True
    )


def show_saved_story_display():
    """Screen 8: View a SINGLE Saved Story (Dynamic)"""

    # --- 1. Check selection ---
    if 'selected_saved_story' not in st.session_state:
        navigate_to('saved_stories_list')
        st.rerun()
        return

    story = st.session_state.selected_saved_story

    # --- Initialize Delete State ---
    if 'confirm_remove_saved' not in st.session_state:
        st.session_state.confirm_remove_saved = False

    # --- CSS ---
    st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #fa5252 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    col_l, col_r = st.columns([10, 1])
    with col_r:
        if st.button("↩", key="back_from_saved_top"):
            st.session_state.confirm_remove_saved = False
            # Return to the LIST, not the profile
            navigate_to('saved_stories_list')
            st.rerun()

    st.write("")

    # --- Dynamic Title & Date ---
    st.markdown(f"<h1 style='text-align: center;'>{story['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #666;'>Saved on {story['date']}</p>", unsafe_allow_html=True)

    # --- Dynamic Content ---
    with st.container(border=True):
        st.markdown(f"""
        <div style='font-size: 1.1em; line-height: 1.6; padding: 20px; background-color: white; border-radius: 10px; min-height: 200px;'>
        {story['content']}
        </div>
        """, unsafe_allow_html=True)

    # --- SHOW IMAGE (New Code) ---
    if story.get("image_path") and os.path.exists(story["image_path"]):
        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_c, col_r = st.columns([2, 2, 2])
        with col_c:
            st.image(story["image_path"], width="stretch")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Buttons (Back & Remove) ---
    c_left, c_center, c_right = st.columns([1, 1, 1])

    with c_center:
        if st.button("Back", key="btn_back_bottom", width="stretch"):
            st.session_state.confirm_remove_saved = False
            navigate_to('saved_stories_list')
            st.rerun()

    with c_right:
        if not st.session_state.confirm_remove_saved:
            if st.button("Remove Story", key="btn_remove_init", type="primary", width="stretch"):
                st.session_state.confirm_remove_saved = True
                st.rerun()

        else:
            st.markdown(
                "<div style='text-align: center; color: #d9534f; font-weight: bold; margin-bottom: 5px; font-size: 0.8em;'>Delete?</div>",
                unsafe_allow_html=True
            )
            sub_yes, sub_cancel = st.columns(2)

            with sub_yes:
                if st.button("Yes", key="btn_remove_yes", type="primary", width="stretch"):
                    # --- DELETE LOGIC ---
                    # Rebuild list excluding this story ID
                    st.session_state.saved_stories = [
                        s for s in st.session_state.saved_stories
                        if s['id'] != story['id']
                    ]

                    st.success("Removed!")
                    time.sleep(2)
                    st.session_state.confirm_remove_saved = False
                    navigate_to('saved_stories_list')
                    st.rerun()

            with sub_cancel:
                if st.button("No", key="btn_remove_cancel", width="stretch"):
                    st.session_state.confirm_remove_saved = False
                    st.rerun()


def show_saved_stories_list():
    """Screen: List of saved stories for the active child (Fixed Image Centering)"""

    # --- 1. Get Current Child ---
    if 'selected_child' not in st.session_state:
        st.error("No child selected.")
        if st.button("Go Back"):
            navigate_to('Manage_Profiles')
            st.rerun()
        return

    child = st.session_state.selected_child

    # --- 2. Filter Stories for THIS Child ---
    child_stories = [
        s for s in st.session_state.saved_stories
        if s['child_id'] == child['ID']
    ]

    # --- TOP BAR ---
    top_left, top_center, top_right = st.columns([1, 6, 1])
    with top_left:
        st.image("images/menu_bar.png", width=40)
    with top_right:
        if st.button("↩", key="back_from_list", type="secondary"):
            navigate_to('child_profile')
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --- HEADER ---
    st.markdown(f"<h1 style='text-align:center;'>{child['Full Name']}'s Library</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAIN CONTENT ---

    # CASE A: NO STORIES FOUND
    if not child_stories:
        st.info(f"No saved stories found for {child['Full Name']} yet.")

        # Just the Create button here (Image moved to bottom)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<h3 style='text-align:center;'>Time to create the first one!</h3>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Create New Story", width="stretch", type="primary"):
                navigate_to('create_new_story')
                st.rerun()

    # CASE B: STORIES EXIST (Show List of Buttons)
    else:
        st.markdown(
            f"<p style='text-align:center; color:#666;'>Found {len(child_stories)} stories</p>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        for story in child_stories:
            # Layout: Spacer | Button | Spacer
            # [1, 3, 1] makes the button take 60% of width and centers it
            c_left, c_btn, c_right = st.columns([1, 3, 1])

            with c_btn:
                # Format: Title | Date
                btn_label = f"{story['title']}  ({story['date']})"

                # If clicked -> Save selection and go to display
                if st.button(btn_label, key=f"story_btn_{story['id']}", width="stretch"):
                    st.session_state.selected_saved_story = story
                    navigate_to('saved_story_display')
                    st.rerun()

            # Small vertical space between buttons
            st.markdown("<div style='margin-bottom: 2px;'></div>", unsafe_allow_html=True)

    # --- BOTTOM SECTION ---
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 1. Back to Profile Button
    b_left, b_center, b_right = st.columns([1.5, 2, 1.5])
    with b_center:
        if st.button("Back to Profile", key="btn_back_main", type="secondary", width="stretch"):
            navigate_to('child_profile')
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Books Image (CENTERED FIX)
    # We use [3, 1, 3] to make the middle column very narrow, forcing the image to center.
    img_left, img_center, img_right = st.columns([3, 1, 3])
    with img_center:
        try:
            st.image("images/books_stack.png", width=150)
        except:
            pass

    # --- Help Icon ---
    sub_col1, sub_col2, sub_col3 = st.columns([1, 6, 1])
    with sub_col1:
        st.markdown(
            '<div style="bottom: 20px; left: 20px; background-color:#e5987d; color:white; width:40px; height:40px; border-radius:50%; text-align:center; line-height:40px; font-weight:bold; cursor:help;">?</div>',
            unsafe_allow_html=True
        )


# --- 3. Main Routing Logic ---
if st.session_state.page == 'login':
    show_login()
elif st.session_state.page == 'Manage_Profiles':
    show_Manage_Profiles()
elif st.session_state.page == 'new_profile':
    show_new_profile()
elif st.session_state.page == 'child_profile':
    show_child_profile()
elif st.session_state.page == "create_new_story":
    show_create_new_story()
elif st.session_state.page == "new_school_story":
    show_new_school_story()
elif st.session_state.page == "new_pet_story":
    show_new_pet_story()
elif st.session_state.page == "new_baby_story":
    show_new_baby_story()
elif st.session_state.page == "custom_story_details":
    show_custom_story_details()
elif st.session_state.page == "additional_details_1":
    show_additional_details_1()
elif st.session_state.page == "additional_details_2":
    show_additional_details_2()
elif st.session_state.page == 'like_story':
    show_like_story()
elif st.session_state.page == 'dislike_story':
    show_dislike_story()
elif st.session_state.page == 'story_display':
    show_story_display()
elif st.session_state.page == 'saved_story_display':
    show_saved_story_display()
elif st.session_state.page == 'saved_stories_list':
    show_saved_stories_list()
