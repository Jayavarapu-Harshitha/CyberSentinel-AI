from sklearn.neural_network import MLPClassifier
def build(seed=42):
    # Standalone CPU neural network with backpropagation and early stopping.
    return MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam',batch_size=256,learning_rate_init=0.001,max_iter=100,early_stopping=True,validation_fraction=0.1,n_iter_no_change=8,random_state=seed)
