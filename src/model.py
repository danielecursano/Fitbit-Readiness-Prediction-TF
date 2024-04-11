from tensorflow.keras.models import load_model

class Model:
    def __init__(self, path, input_shape):
        try:
            self.model = load_model(path)
        except Exception as e:
            raise e
        self.input_shape = input_shape
        if path.split(".")[-1] == "h5":
            print("model compiled successfully")
            self.model.compile(optimizer="Adam")

    def predict(self, vector):
        assert vector.shape == self.input_shape
        return self.model.predict(vector)

if __name__ == "__main__":
    model = Model("/Users/daniele/Desktop/fitbit/data/fitbit-model.h5", (1, 11))
    import numpy as np
    inp = np.array([6.00000000e+00, 1.20847000e+02, 5.11690000e+01,
                    3.64100000e+00,9.61000000e+01, 7.80000000e+01,
                    2.10000000e+01, 9.30000000e+01,5.40000000e+01,
                    7.12008502e-02, 7.50000000e+01]).reshape((1, 11))
    print(model.predict(inp))
