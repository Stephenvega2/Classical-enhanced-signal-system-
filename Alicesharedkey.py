import numpy as np
import hashlib
from Crypto.Cipher import AES
import base64
import json

def generate_aes256_key_from_params(params):
    param_bytes = np.array(params).tobytes()
    return hashlib.sha256(param_bytes).digest()

def encrypt_aes256(key, data):
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return cipher.nonce, ciphertext, tag

def run_classical_optimization():
    optimal_params = np.array([0.12345, 0.12345, 0.12345, 0.12345])
    return optimal_params

def main():
    optimal_params = run_classical_optimization()
    print(f"Alice's optimal parameters: {optimal_params}")

    aes_key = generate_aes256_key_from_params(optimal_params)
    message = "Hello Bob, this is Alice. Only you should be able to read this!"
    nonce, ciphertext, tag = encrypt_aes256(aes_key, message)
    print(f"Alice's encrypted message (hex): {ciphertext.hex()}")

    # Prepare for JSON (convert bytes to base64 strings)
    data = {
        "optimal_params": optimal_params.tolist(),
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "tag": base64.b64encode(tag).decode()
    }

    with open("for_bob.json", "w") as f:
        json.dump(data, f)
    print("Alice has written the ciphertext and parameters to for_bob.json")

if __name__ == "__main__":
    main()
