# GeminiGradioApp

## Introduction
GeminiGradioApp is an interactive application that utilizes Google's Gemini Pro and Gemini Pro Vision APIs. This application presents the fundamental features of AI in areas such as natural language processing and image recognition through a user-friendly interface. Additionally, it allows users to experience multimodal AI interactions.

## Features
- **Interactive AI Chat**: Experience real-time conversations powered by Gemini's advanced AI technology.
- **Image Processing Capabilities**: Explore image recognition features with Gemini Pro Vision.
- **Customizable Interaction**: Adjust settings like temperature and token limits to tailor your AI interactions.

## Accessing the Gradio Web Interface
[Gradio Web Interface](https://huggingface.co/spaces/meryem-sakin/GeminiApp)

### Prerequisites
- Python 3.10
- Google-generativeai
- Gradio
- Pillow

### Installation

Clone the repository and install the required packages:

1. Install Conda and create a new environment:

    ```bash
    conda create --name venv python=3.9.16
    conda activate venv
    ```

2. Clone the repository:

    ```bash
    git clone https://github.com/meryemsakin/GeminiGradioApp.git
    ```

3. Navigate to the project's root directory:

    ```bash
    cd GeminiGradioApp
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Running the Application:

    ```bash
    python app.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
