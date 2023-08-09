# Import necessary libraries
import pandas as pd  # Library for data manipulation and analysis
from pdfminer.high_level import extract_text  # Function for extracting text from PDF files
from nltk.translate import bleu_score  # Library for BLEU score calculation
from nltk.tokenize import sent_tokenize  # Function for sentence tokenization
import re  # Library for regular expressions
import nltk  # Natural Language Toolkit for NLP functions

# Download the required data for sentence tokenization
nltk.download('punkt')

# Define the main function
def main():
    """
    Main function to extract paragraphs from English and Chinese texts,
    match them using BLEU score, and export results to CSV and Excel files.
    """

    # Extracting the english and chinese content from files
    english_pdf_text = extract_text("english.pdf")
    chinese_pdf_text = extract_text("chinese.pdf")

    # Split English text into paragraphs
    english_paragraphs = re.split(r'\n\n+|\r\n\r\n+', english_pdf_text)

    # Split Chinese text into paragraphs
    chinese_paragraphs = re.split(r'\n\n+|\r\n\r\n+', chinese_pdf_text)

    # Initialize a list to store paired paragraphs
    paired_paragraphs = []

    # Iterate through English paragraphs and find matching Chinese paragraphs
    for _, eng_paragraph in enumerate(english_paragraphs):
        # Tokenize English paragraph into sentences
        eng_sentences = sent_tokenize(eng_paragraph)
        best_bleu_score = 0.0
        best_chi_paragraph = None

        # Iterate through Chinese paragraphs to find the best matching one
        for chi_paragraph in chinese_paragraphs:
            # Tokenize Chinese paragraph into sentences
            chi_sentences = sent_tokenize(chi_paragraph)
            bleu_scores = []

            # Calculate BLEU score for each English sentence against Chinese sentences
            for eng_sentence in eng_sentences:
                sentence_bleu_scores = [
                    bleu_score.sentence_bleu([eng_sentence.split()], chi_sentence.split())
                    for chi_sentence in chi_sentences
                ]
                # Get the maximum BLEU score among Chinese sentences for the current English sentence
                max_bleu_score_per_sentence = max(sentence_bleu_scores) if sentence_bleu_scores else 0
                bleu_scores.append(max_bleu_score_per_sentence)

            # Calculate the average BLEU score for the Chinese paragraph
            avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

            # Update the best matching Chinese paragraph if the current one has a higher BLEU score
            if avg_bleu_score > best_bleu_score:
                best_bleu_score = avg_bleu_score
                best_chi_paragraph = chi_paragraph

        # If a matching Chinese paragraph is found, add the pair to the list
        if best_chi_paragraph:
            paired_paragraphs.append((eng_paragraph, best_chi_paragraph))

    # Create a DataFrame with paired paragraphs and export to CSV and Excel files
    df = pd.DataFrame(paired_paragraphs, columns=["English Paragraph", "Chinese Paragraph"])
    df.to_csv("output.csv", index=False)
    df.to_excel("output.xlsx", index=False)
    print("Data exported to 'paired_paragraphs.csv' and 'paired_paragraphs.xlsx'")

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
