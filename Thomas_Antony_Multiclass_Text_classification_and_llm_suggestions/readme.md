
# Text Multi Class Classification Project

This project aims to classify and predict various issues based on full user complaint descriptions. Leveraging advanced NLP techniques and pre-trained transformer models, this solution helps streamline the process of categorizing and addressing user issues.

## Features
- **Multi-Class Text Classification**: Predicts different issue categories based on complaint text.
- **Streamlit Interface**: Simple and interactive web interface for entering complaints and viewing predictions.
- **Transformer Models**: Uses state-of-the-art transformer-based models, including Llama 2 from Hugging Face, for high-accuracy classification.
- **End-to-End Workflow**: From data preprocessing to prediction, the project is designed for scalability and efficiency.


## Disclaimer
Note: This repository does not provide any pre-trained models. You will need to manually download the Llama 2 model from Hugging Face and place it in a folder named models within the main project directory.

## Requirements

To run this project, install the following dependencies (see `requirements.txt` for more details):

- **ctransformers**
- **langchain**
- **python-box**
- **streamlit**
- **sentence-transformers**
- **uvicorn**
- **langchain-community**
- **langchain-core**
- **ipykernel**
- **tensorflow==2.17.0**
- **scikit-learn**
- **transformers**

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Using Llama 2 Model

This project utilizes **Llama 2**, an open-source large language model from Hugging Face, for enhanced text classification. The model has been downloaded to the local system for faster and more efficient processing.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/text-classification-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd text-classification-project
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the Llama 2 model is accessible in your local system and configured for use in `app.py`.
5. Start the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Launch the app with Streamlit and enter complaint text into the input field to receive a prediction on the issue type.
- Explore different complaint categories and view the model's confidence scores for each classification.

## Project Structure

- `app.py`: Main file for the Streamlit interface.
- `models/`: Contains pre-trained transformer models, including Llama 2.
- `utils/`: Helper functions for data preprocessing and prediction.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: Overview and setup instructions for the project.

## Technologies Used

- **TensorFlow** for model building and fine-tuning.
- **Transformers** library for accessing pre-trained models.
- **LangChain** and **Sentence-Transformers** for enhanced language model capabilities.
- **Streamlit** for deploying a user-friendly web application.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

This README now includes the details of the Llama 2 model setup and usage. Let me know if there's anything else you'd like to add!