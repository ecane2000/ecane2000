import spacy
import os

def main():
    # Get the full path to the model within the virtual environment
    model_path = os.path.join(os.path.dirname(spacy.__file__), 'models', 'en_core_web_sm', '__init__.py')
    
    # Load spaCy language model
    nlp = spacy.load(model_path)

    # Your corpus processing logic here
    # ...

if __name__ == "__main__":
    main()
