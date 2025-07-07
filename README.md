# Voice Age and Emotion Detector

A machine learning application that analyzes male voices to predict age and, for senior citizens, their emotional state. The application features a simple graphical user interface (GUI) for ease of use.

## Core Logic

1.  **Input**: The user uploads a voice recording (`.wav`, `.mp3`).
2.  **Gender Check**: The model first determines the speaker's gender.
    - If a **female voice** is detected, the program stops and displays: "**Upload male voice.**"
3.  **Age Prediction**:
    - If a **male voice** is detected, the model predicts the speaker's age.
4.  **Emotion Detection (Conditional)**:
    - If the predicted age is **below 60**, only the age is displayed.
    - If the predicted age is **60 or above**, the person is marked as a **"Senior Citizen,"** and the model also detects their emotion (e.g., happy, sad, neutral).

## Features

-   **Gender-Specific Processing**: Exclusively analyzes male voices.
-   **Age Prediction**: A regression model to estimate the speaker's age.
-   **Emotion Classification**: A classification model for senior speakers.
-   **Simple GUI**: An intuitive interface to upload a file and see the results.

## How to Run

**1. Prerequisites**
- Python 3.7+
- A virtual environment is recommended.

**2. Installation**

Clone the repository and install the required packages:
```bash
git clone https://github.com/your-username/voice-age-emotion-detector.git
cd voice-age-emotion-detector
pip install -r requirements.txt