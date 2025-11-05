import sys
print(sys.executable)
print(sys.path)
try:
    import numpy
    print(numpy.__version__, numpy.__file__)
except Exception as e:
    print(f"Error: {e}")