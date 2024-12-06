import time
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# Placeholder for a McEliece encryption module simulation
class McEliece:
    def generate_keypair(self):
        # Simulate key generation delay
        time.sleep(0.005)  # Simulate 5 ms processing time
        return "public_key", "private_key"

    def encrypt(self, message, public_key):
        # Simulate encryption delay for McEliece
        time.sleep(0.002)  # Simulate 2 ms processing time
        return "encrypted_message"

# Simulate AES-256 encryption
def aes_encryption(message, key):
    padded_info = pad(message, AES.block_size)
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(padded_info)

# Simulate McEliece encryption
def mceliece_encryption(message):
    mceliece_cipher = McEliece()
    public_key, _ = mceliece_cipher.generate_keypair()
    return mceliece_cipher.encrypt(message, public_key)

# Test encryption times
def test_encryption_times():
    message = b"A" * 64  # 512-bit message
    aes_key = b"AESKeyForTest128"  # 16-byte key

    # AES-256 timing
    start_aes = time.perf_counter()
    aes_encryption(message, aes_key)
    end_aes = time.perf_counter()

    # McEliece timing
    start_mc = time.perf_counter()
    mceliece_encryption(message)
    end_mc = time.perf_counter()

    aes_ms = (end_aes - start_aes) * 1000
    mc_ms = (end_mc - start_mc) * 1000

    print("AES-256 Encryption Time (ms): {:.3f}".format(aes_ms))
    print("McEliece Encryption Time (ms): {:.3f}".format(mc_ms))


# Run the test
if __name__ == "__main__":
    test_encryption_times()

