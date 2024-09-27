import numpy as np
import qiskit 
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit.library import ZZFeatureMap
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def load_twitter_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['text'])
    X = df['text'].values
    y = df['sentiment'].values
    return X, y

def preprocess_data(X, y, max_features=100):
    y = (y == 'positive').astype(int)
    
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(X)
    
    if X.shape[1] < max_features:
        padding = np.zeros((X.shape[0], max_features - X.shape[1]))
        X = np.hstack([X.toarray(), padding])
    else:
        X = X.toarray()
    
    # Normalize each sample
    X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
    
    return X, y, vectorizer

def quantum_conv_layer(qc, qubits, params):
    for i in range(len(qubits) - 1):
        qc.rzz(params[i], qubits[i], qubits[i+1])
    return qc

def quantum_pool_layer(qc, qubits):
    for i in range(0, len(qubits) - 1, 2):
        qc.cx(qubits[i], qubits[i+1])
        qc.measure(qubits[i+1], i//2)
    return qc

def create_quantum_circuit(num_qubits, params):
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits // 2)
    qc = QuantumCircuit(qr, cr)
    
    feature_map = ZZFeatureMap(num_qubits)
    qc.append(feature_map, qr)
    
    qc = quantum_conv_layer(qc, qr, params[:num_qubits-1])
    qc = quantum_pool_layer(qc, qr)
    
    return qc

class QuantumCNN:
    def __init__(self, num_qubits, shots=1000):
        self.num_qubits = num_qubits
        self.shots = shots
        self.params = np.random.randn(num_qubits - 1)
    
    def get_expectation(self, x):
        qc = create_quantum_circuit(self.num_qubits, self.params)
        qc.initialize(x, qc.qubits)
        
        backend = AerSimulator()
        job = execute(qc, backend, shots=self.shots)
        result = job.result().get_counts(qc)
        
        expectation = sum([int(bitstring, 2) * count for bitstring, count in result.items()]) / self.shots
        return expectation
    
    def predict(self, X):
        return np.array([self.get_expectation(x) for x in X])

def train_quantum_cnn(X_train, y_train, num_qubits, epochs=50, learning_rate=0.01):
    model = QuantumCNN(num_qubits)
    
    for epoch in range(epochs):
        y_pred = model.predict(X_train)
        loss = np.mean((y_pred - y_train) ** 2)
        
        grad = np.zeros_like(model.params)
        epsilon = 1e-6
        for i in range(len(model.params)):
            model.params[i] += epsilon
            y_pred_plus = model.predict(X_train)
            model.params[i] -= 2 * epsilon
            y_pred_minus = model.predict(X_train)
            model.params[i] += epsilon
            
            grad[i] = np.mean((y_pred_plus - y_pred_minus) * (y_train - y_pred)) / (2 * epsilon)
        
        model.params -= learning_rate * grad
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return model

if __name__ == "__main__":
    X, y = load_twitter_data(r"twittersentiment.csv")
    X, y, vectorizer = preprocess_data(X, y, max_features=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_qubits = 8
    model = train_quantum_cnn(X_train, y_train, num_qubits, epochs=100)

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    print(f"Test Accuracy: {accuracy:.4f}")