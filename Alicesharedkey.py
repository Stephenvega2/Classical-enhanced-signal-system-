import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import pickle

# Key derivation from optimal parameters
def generate_aes256_key_from_params(params):
    param_bytes = np.array(params).tobytes()
    return hashlib.sha256(param_bytes).digest()

def encrypt_aes256(key, data):
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return cipher.nonce, ciphertext, tag

def run_classical_optimization():
    # This is a minimal classical optimization like your Signalsystem.py
    optimal_params = np.array([0.12345, 0.12345, 0.12345, 0.12345])  # Example; use your logic here if you want
    return optimal_params

def main():
    # Step 1: Alice runs her optimization
    optimal_params = run_classical_optimization()
    print(f"Alice's optimal parameters: {optimal_params}")

    # Step 2: Alice derives the key
    aes_key = generate_aes256_key_from_params(optimal_params)

    # Step 3: Alice encrypts her message
    message = "Hello Bob, this is Alice. Only you should be able to read this!"
    nonce, ciphertext, tag = encrypt_aes256(aes_key, message)
    print(f"Alice's encrypted message (hex): {ciphertext.hex()}")

    # Step 4: Alice saves everything Bob needs
    with open("for_bob.pkl", "wb") as f:
        pickle.dump({
            "optimal_params": optimal_params,
            "nonce": nonce,
            "ciphertext": ciphertext,
            "tag": tag
        }, f)
    print("Alice has written the ciphertext and parameters to for_bob.pkl")

if __name__ == "__main__":
    main()
