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
        return self.model.predict(vector)
    
    def train(self, x_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    def save(self, filepath):
        self.model.save(filepath)

if __name__ == "__main__":
    model = Model("/Users/daniele/Desktop/fitbit/data/fitbit-model.h5", (1, 11))
    import numpy as np
    inp = np.array([6.00000000e+00, 1.20847000e+02, 5.11690000e+01,
                    3.64100000e+00,9.61000000e+01, 7.80000000e+01,
                    2.10000000e+01, 9.30000000e+01,5.40000000e+01,
                    7.12008502e-02, 7.50000000e+01]).reshape((1, 11))
    out = model.predict(inp)
    out = max(max(out, 0), min(out, 100))
    print(out)
