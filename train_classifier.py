import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """Load data from a pickle file"""
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def preprocess_data(data_dict):
    """Preprocess data by padding or truncating lists to the maximum length"""
    data = []
    labels = np.asarray(data_dict['labels'])
    max_features = max(len(item) for item in data_dict['data'])

    for item in data_dict['data']:
        if not isinstance(item, np.ndarray):
            if len(item) == 1:
                item = np.full((max_features,), item[0])
            else:
                item = np.pad(item, (0, max_features - len(item)), mode='constant')
        data.append(item)

    return np.array(data), labels
    return np.asarray(data), labels

def train_model(data, labels):
    """Train a random forest classifier on the data"""
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model, x_test, y_test

def evaluate_model(model, x_test, y_test):
    """Evaluate the model on the test data"""
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print('{}% of samples were classified correctly !'.format(score * 100))

def save_model(model, file_path):
    """Save the model to a pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model}, f)

if __name__ == '__main__':
    data_dict = load_data('./data.pickle')
    data, labels = preprocess_data(data_dict)
    model, x_test, y_test = train_model(data, labels)
    evaluate_model(model, x_test, y_test)
    save_model(model, 'model.p')