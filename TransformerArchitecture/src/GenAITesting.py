import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "classes"))

from classes.class_inference import Inference

print(Inference.Chat("This is the input", 5))