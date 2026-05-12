from deepface import DeepFace
import os

print("Testing DeepFace load...")
try:
    # Try to load a model to force download
    DeepFace.build_model("Facenet512")
    print("Facenet512 loaded!")
except Exception as e:
    print(f"Error: {e}")
