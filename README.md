------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Spam Email Classifier (Naïve Bayes, TF-IDF, and OCR Support)

This project implements a complete spam email classification system using Python, scikit-learn, TF-IDF feature extraction, and a Multinomial Naïve Bayes classifier.
The application includes a command-line interface that allows users to:

    1. Paste email text for classification

    2. Provide a screenshot image of an email to be classified via OCR

    3. Exit the program

The goal of this project is to demonstrate how traditional machine learning methods can be applied to spam detection, while also providing a practical interface for safe, manual testing.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

FEATURES

- TF-IDF vectorization for text preprocessing

- Multinomial Naïve Bayes classifier

- Train/test split with stratification

- Five-fold cross-validation

- Performance metrics: accuracy, precision, recall, F1-score

- Screenshot classification using Tesseract OCR

- Safe, offline command-line interface (no inbox access)

- Simple and extensible codebase

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Project Structure

SpamClassifier/

    -spam_classifier.py        # Main executable script
    -README.md                 # Project documentation

        --data/
            ---emails.csv            # Dataset used for training and evaluation

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

REQUIREMENTS:

Python Version:
--------------

Python 3.10 or later is recommended.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Python Packages:
---------------

Install required packages using:

pip install scikit-learn pandas numpy nltk pillow pytesseract


Download NLTK stopwords (required for preprocessing):

python -c "import nltk; nltk.download('stopwords')"

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Tesseract OCR (for screenshot classification)

Tesseract must be installed for OCR functionality.

Windows:

Download from:
https://github.com/UB-Mannheim/tesseract/wiki

Typical installation path:

C:\Users\<USERNAME>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe

macOS:

brew install tesseract

Linux (Debian/Ubuntu):

sudo apt-get install tesseract-ocr

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Configuring Tesseract Path (Windows)
-----------------------------------

To enable screenshot OCR on Windows, specify the Tesseract executable path in spam_classifier.py immediately after importing pytesseract:

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\<USERNAME>\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

Replace <USERNAME> with your Windows user name.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Dataset Format
--------------

The classifier requires a CSV file named emails.csv inside the data/ directory.
The CSV must contain two columns:

label,text
Where:

label is either spam or ham

text is the email body

Example:

label,text
spam,"You have won a free gift card. Click here to claim it."
ham,"Can you send me the meeting notes when you get a chance?"

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Running the Classifier
----------------------

Navigate to the project directory:

cd SpamClassifier

Run the program:

python spam_classifier.py


You will see the following interface:

==============================
  Spam Email Classifier (NB)
==============================

Choose an option:
  1) Classify pasted email text
  2) Classify screenshot image (OCR)
  3) Exit

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Example Usage
-------------

Option 1: Classify Pasted Text

Paste an email message, then press Enter on a blank line to finish:

Dear Customer,
Your account has been temporarily locked. Please verify your identity...


The program outputs:

Prediction: SPAM
Spam probability: 0.87


Option 2: Classify a Screenshot

Enter a valid file path:

C:\Users\Name\Desktop\EmailScreenshot.png


The program will extract the text using OCR and classify it automatically.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Evaluation
----------

The script prints a full set of evaluation metrics, including:

    - Accuracy

    - Precision

    - Recall

    - F1-Score

    - Classification report

    - Five-fold cross-validation accuracy

This provides a comprehensive view of model performance.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Limitations
-----------

    - Small datasets may lead to unstable predictions

    - OCR accuracy depends on screenshot clarity and formatting

    - This tool does not connect to any email accounts for safety reasons

    - Designed for educational and demonstration purposes

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Future Enhancements
-------------------

Potential extensions include:

    - Word embeddings (Word2Vec, GloVe, BERT)

    - Neural network models

    - Confusion matrix and visualizations

    - Web interface (Flask or FastAPI)

    - Model saving and loading

    - Larger training datasets

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

License
-------

This project is distributed under the MIT License.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
