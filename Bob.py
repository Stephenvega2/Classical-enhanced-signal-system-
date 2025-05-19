import numpy as np
import hashlib
from Crypto.Cipher import AES
import base64
import json
import os

def generate_aes256_key_from_params(params):
    param_bytes = np.array(params).tobytes()
    return hashlib.sha256(param_bytes).digest()

def decrypt_aes256(key, nonce, ciphertext, tag):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')

def main():
    if not os.path.exists("for_bob.json"):
        print("Waiting for Alice... file not found.")
        return

    with open("for_bob.json", "r") as f:
        data = json.load(f)

    optimal_params = np.array(data["optimal_params"])
    nonce = base64.b64decode(data["nonce"])
    ciphertext = base64.b64decode(data["ciphertext"])
    tag = base64.b64decode(data["tag"])

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
