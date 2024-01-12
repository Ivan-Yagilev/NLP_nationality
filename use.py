import transliterate
import torch
from argparse import Namespace
from classes import SurnameClassifier, SurnameDataset


args = Namespace(
    # Пути к директориям
    surname_csv="data/surnames/surnames_with_splits.csv",
    vectorizer_file="model_storage/ch4/surname_mlp/vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage/ch4/surname_mlp",
    # Гиперпараметры модели
    hidden_dim=300
)


def predict_nationality(surname, classifier, vectorizer):
    vectorized_surname = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).view(1, -1)
    result = classifier(vectorized_surname, apply_softmax=True)

    probability_values, indices = result.max(dim=1)
    index = indices.item()

    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
    probability_value = probability_values.item()

    return {'nationality': predicted_nationality, 'probability': probability_value}


def predict(surnameKyrillic):
    new_surname = transliterate.translit(surnameKyrillic, reversed=True)
    dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv,
                                                              args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()

    classifier = SurnameClassifier(input_dim=len(vectorizer.surname_vocab), 
                                hidden_dim=args.hidden_dim, 
                                output_dim=len(vectorizer.nationality_vocab))

    classifier.load_state_dict(torch.load("model_storage/ch4/surname_mlp/model.pth"))
    classifier.eval()

    classifier = classifier.to("cpu")
    prediction = predict_nationality(new_surname, classifier, vectorizer)
    return prediction['nationality'], prediction['probability']