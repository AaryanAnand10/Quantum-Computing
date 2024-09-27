import cirq
import numpy as np
import time
import threading
import hashlib

class QuantumOTPGenerator:
    def __init__(self, num_qubits=20):
        """
        Initializes the Quantum OTP Generator.

        Args:
            num_qubits (int): Number of qubits to use for generating randomness.
                             More qubits allow for generating larger OTPs.
        """
        self.num_qubits = num_qubits

    def generate_otp(self, otp_length=6):
        """
        Generates a numerical OTP using quantum randomness.

        Args:
            otp_length (int): The length of the OTP to generate.

        Returns:
            str: A numerical OTP of specified length.
        """
        qubits = cirq.LineQubit.range(self.num_qubits)
        circuit = cirq.Circuit()

        # Apply Hadamard gates to create superposition
        for q in qubits:
            circuit.append(cirq.H(q))

        # Measure all qubits
        circuit.append(cirq.measure(*qubits, key='result'))

        # Run the circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
        measured_bits = result.measurements['result'][0]

        # Convert bits to integer
        random_number = int(''.join(str(bit) for bit in measured_bits), 2)

        # Ensure the OTP is of the desired length
        otp = str(random_number % (10 ** otp_length)).zfill(otp_length)
        return otp

class UserDatabase:
    def __init__(self):
        """
        Initializes the user database.
        """
        self.users = {}

    def register_user(self, username, password):
        """
        Registers a new user with a hashed password.

        Args:
            username (str): The username.
            password (str): The plaintext password.

        Returns:
            bool: True if registration is successful, False if user already exists.
        """
        if username in self.users:
            return False
        hashed_password = self._hash_password(password)
        self.users[username] = hashed_password
        return True

    def verify_user(self, username, password):
        """
        Verifies user credentials.

        Args:
            username (str): The username.
            password (str): The plaintext password.

        Returns:
            bool: True if credentials are correct, False otherwise.
        """
        if username not in self.users:
            return False
        hashed_password = self._hash_password(password)
        return self.users[username] == hashed_password

    def _hash_password(self, password):
        """
        Hashes the password using SHA-256.

        Args:
            password (str): The plaintext password.

        Returns:
            str: The hexadecimal hash of the password.
        """
        return hashlib.sha256(password.encode()).hexdigest()

class LoginSystem:
    def __init__(self, otp_length=6, otp_validity=300):
        """
        Initializes the Login System.

        Args:
            otp_length (int): Length of the OTP.
            otp_validity (int): OTP validity period in seconds.
        """
        self.user_db = UserDatabase()
        self.otp_generator = QuantumOTPGenerator()
        self.otp_length = otp_length
        self.otp_validity = otp_validity  # in seconds
        self.active_otps = {}  # Stores OTPs with their expiry times

    def register(self, username, password):
        """
        Registers a new user.

        Args:
            username (str): The username.
            password (str): The plaintext password.

        Returns:
            str: Success or failure message.
        """
        success = self.user_db.register_user(username, password)
        if success:
            return "Registration successful."
        else:
            return "Username already exists."

    def login(self, username, password):
        """
        Initiates the login process by verifying credentials and generating OTP.

        Args:
            username (str): The username.
            password (str): The plaintext password.

        Returns:
            str: OTP generation status or error message.
        """
        if not self.user_db.verify_user(username, password):
            return "Invalid username or password."

        # Generate OTP
        otp = self.otp_generator.generate_otp(self.otp_length)
        self.active_otps[username] = {
            'otp': otp,
            'expiry': time.time() + self.otp_validity
        }

        # Simulate sending OTP (e.g., via email/SMS)
        # Here, we'll just print it
        print(f"OTP for {username}: {otp} (valid for {self.otp_validity} seconds)")

        # Start a timer to invalidate the OTP after its validity period
        threading.Thread(target=self._invalidate_otp, args=(username,), daemon=True).start()

        return "OTP has been sent."

    def verify_otp(self, username, otp_input):
        """
        Verifies the provided OTP.

        Args:
            username (str): The username.
            otp_input (str): The OTP entered by the user.

        Returns:
            str: Verification result message.
        """
        if username not in self.active_otps:
            return "No active OTP. Please initiate login again."

        otp_info = self.active_otps[username]
        if time.time() > otp_info['expiry']:
            del self.active_otps[username]
            return "OTP has expired. Please initiate login again."

        if otp_input == otp_info['otp']:
            del self.active_otps[username]
            return "Login successful!"
        else:
            return "Invalid OTP."

    def _invalidate_otp(self, username):
        """
        Invalidates the OTP after its validity period.

        Args:
            username (str): The username.
        """
        time.sleep(self.otp_validity)
        if username in self.active_otps:
            del self.active_otps[username]

def main():
    login_system = LoginSystem()

    while True:
        print("\n--- Quantum OTP Login System ---")
        print("1. Register")
        print("2. Login")
        print("3. Exit")
        choice = input("Select an option: ")

        if choice == '1':
            username = input("Enter desired username: ")
            password = input("Enter desired password: ")
            message = login_system.register(username, password)
            print(message)

        elif choice == '2':
            username = input("Enter username: ")
            password = input("Enter password: ")
            login_message = login_system.login(username, password)
            print(login_message)

            if login_message == "OTP has been sent.":
                otp_input = input("Enter the OTP: ")
                verification_message = login_system.verify_otp(username, otp_input)
                print(verification_message)

        elif choice == '3':
            print("Exiting the system. Goodbye!")
            break

        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
