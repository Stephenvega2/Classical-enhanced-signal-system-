import numpy as np
import hashlib
from Crypto.Cipher import AES
import pickle
import os

def generate_aes256_key_from_params(params):
    param_bytes = np.array(params).tobytes()
    return hashlib.sha256(param_bytes).digest()

def decrypt_aes256(key, nonce, ciphertext, tag):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')

def main():
    if not os.path.exists("for_bob.pkl"):
        print("Waiting for Alice... file not found.")
        return

    with open("for_bob.pkl", "rb") as f:
        bundle = pickle.load(f)

    optimal_params = bundle["optimal_params"]
    nonce = bundle["nonce"]
    ciphertext = bundle["ciphertext"]
    tag = bundle["tag"]

    print(f"Bob's received optimal parameters: {optimal_params}")

    aes_key = generate_aes256_key_from_params(optimal_params)

    try:
        decrypted_message = decrypt_aes256(aes_key, nonce, ciphertext, tag)
        print(f"Bob successfully decrypted: {decrypted_message}")
    except Exception as e:
        print("Decryption failed! Possible tampering or key mismatch.")
        print(e)

if __name__ == "__main__":
    main()
