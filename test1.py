import cirq
import numpy as np
import time
import threading

class QuantumChatApp:
    def __init__(self, num_qubits=10):
        self.num_qubits = num_qubits
        self.shared_key = None
        self.messages = []

    def generate_quantum_key(self):
        # Alice generates random bits and bases
        alice_bits = np.random.randint(2, size=self.num_qubits)
        alice_bases = np.random.randint(2, size=self.num_qubits)

        # Alice prepares qubits
        qubits = cirq.LineQubit.range(self.num_qubits)
        alice_circuit = cirq.Circuit()
        
        for i in range(self.num_qubits):
            if alice_bases[i] == 0:  # Z basis
                if alice_bits[i] == 1:
                    alice_circuit.append(cirq.X(qubits[i]))
            else:  # X basis (Hadamard)
                if alice_bits[i] == 1:
                    alice_circuit.append(cirq.X(qubits[i]))
                alice_circuit.append(cirq.H(qubits[i]))

        # Bob generates random bases and measures
        bob_bases = np.random.randint(2, size=self.num_qubits)
        bob_circuit = cirq.Circuit()
        
        for i in range(self.num_qubits):
            if bob_bases[i] == 1:  # Measure in X basis
                bob_circuit.append(cirq.H(qubits[i]))
            bob_circuit.append(cirq.measure(qubits[i], key=f'qubit-{i}'))

        # Run the quantum circuit
        simulator = cirq.Simulator()
        final_circuit = alice_circuit + bob_circuit
        result = simulator.run(final_circuit)

        # Extract Bob's measurement results
        bob_bits = np.array([int(result.measurements[f'qubit-{i}']) for i in range(self.num_qubits)])

        # Compare bases and keep only matching results
        matching_bases = alice_bases == bob_bases
        self.shared_key = alice_bits[matching_bases]

    def encrypt_message(self, message):
        if self.shared_key is None:
            raise ValueError("Shared key not generated yet.")
        
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        encrypted_message = ''.join(str(int(b) ^ k) for b, k in zip(binary_message, np.tile(self.shared_key, len(binary_message)//len(self.shared_key) + 1)))
        return encrypted_message

    def decrypt_message(self, encrypted_message):
        if self.shared_key is None:
            raise ValueError("Shared key not generated yet.")
        
        decrypted_binary = ''.join(str(int(b) ^ k) for b, k in zip(encrypted_message, np.tile(self.shared_key, len(encrypted_message)//len(self.shared_key) + 1)))
        decrypted_message = ''.join(chr(int(decrypted_binary[i:i+8], 2)) for i in range(0, len(decrypted_binary), 8))
        return decrypted_message

    def send_message(self, sender, message):
        encrypted_message = self.encrypt_message(message)
        self.messages.append((sender, encrypted_message))

    def display_messages(self):
        for sender, encrypted_message in self.messages:
            decrypted_message = self.decrypt_message(encrypted_message)
            print(f"{sender}: {decrypted_message}")

    def run(self):
        print("Generating quantum key...")
        self.generate_quantum_key()
        print("Quantum key generated!")

        print("\nWelcome to the Quantum Chat App!")
        print("Type 'quit' to exit the chat.")

        def receive_messages():
            while True:
                time.sleep(1)
                if self.messages:
                    print("\nNew message received!")
                    self.display_messages()
                    self.messages.clear()

        # Start the message receiving thread
        threading.Thread(target=receive_messages, daemon=True).start()

        while True:
            message = input("You: ")
            if message.lower() == 'quit':
                break
            self.send_message("You", message)

        print("Chat ended. Goodbye!")

if __name__ == "__main__":
    app = QuantumChatApp()
    app.run()