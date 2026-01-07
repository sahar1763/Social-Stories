# Personalized Social Story Generator
Course project for Intelligent Interactive Systems (00960235): AI-assisted generation of personalized Social Stories

## Overview
This project presents an interactive prototype for generating **personalized Social Stories** for children with autism.  
The system assists parents and therapists by guiding them through a structured input process and using **Large Language Models (LLMs)** to generate supportive, child-friendly stories tailored to individual needs.
Stories can optionally include illustrative visualization based on a user-provided image.

The goal of the project is to explore how intelligent interactive systems can reduce user burden while maintaining alignment with established Social Story principles.

---

## Motivation
Social Stories are widely used to help children with autism understand and prepare for everyday changes.  
However, creating effective and personalized stories requires time, expertise, and careful adaptation to each child.

This project investigates how LLMs, combined with guided user interaction, can support the creation of personalized Social Stories while preserving clarity, consistency, and emotional safety.
All generated stories are designed to follow the well-established guidelines for Social Stories developed by Carol Gray, ensuring alignment with accepted therapeutic principles.
---

## Key Features
- Guided creation of personalized Social Stories
- Child profile creation including developmental and sensory information
- Combination of structured questions and free-text input
- LLM-based story generation
- Optional illustrative image generation
- Web-based interactive interface

---

## System Overview
The system operates in several stages:
1. User and child profile creation
2. Description of a social situation (custom or predefined templates)
3. Guided follow-up questions to refine context
4. Prompt construction based on collected inputs
5. LLM-based generation of a personalized Social Story

This staged interaction is designed to balance personalization with ease of use.

---

## Technologies Used
- Python
- Streamlit
- Large Language Models (LLMs)
- Diffusion-based image generation
- GitHub for version control

---

## Repository Structure
├── Prototype.py # Main Streamlit application
├── images/ # Project integrated images
├── uploads/ # Project dedicated path for saving generated images
├── requirements.txt # Python dependencies
├── README.md # Project documentation


---

## Installation
1. Clone the repository:
git clone https://github.com/sahar1763/Social-Stories.git

2. Install dependencies:
pip install -r requirements.txt
pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118

	### Ollama (Required for Story Generation)
This project uses Ollama as a local LLM backend.
Ollama must be installed separately and running on the host machine.

- Download: https://ollama.com
- Tested model: mistral
- The system works fully offline after installation.
- No data or images are sent outside the local machine.
- Make sure to update OLLAMA_PATH in Prototype.py file

	### GPU Support (Optional)
AI image generation uses Stable Diffusion XL via Diffusers.

- GPU with CUDA support is recommended
- CPU-only mode is supported but slow
- CUDA drivers are not managed by pip


3. Run the application:
streamlit run Prototype.py

4. Sign-up/Sign-in:
On the homepage, no sign-up is required.
You may enter any username and proceed directly to the application.


---

## Usage
1. Create a user account
2. Define a child profile with relevant personal and developmental details
3. Select or describe a social situation
4. Answer guided follow-up questions (optional)
5. Generate and review the Social Story
6. Save or regenerate stories as needed

---

## Ethical and Privacy Considerations
This project is a research prototype developed for academic purposes.  
All data is processed locally, and no personal information is intentionally stored or shared externally.

The system is not intended to replace professional therapeutic judgment and should be used as a supportive tool only.

---

## Limitations
- The system is a prototype and has not been clinically validated
- Generated stories may require human review and adjustment
- Image generation capabilities are limited and optional
- Story quality depends on the provided inputs and model behavior

---

## Course Information
Course: **Intelligent Interactive Systems (00960235)**  
Institution: Technion  
Project Type: Final Course Project

---

## Author
Sahar Cohen
Maayan Farbstein
