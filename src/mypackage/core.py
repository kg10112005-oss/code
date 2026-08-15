def program1():
    """Runs PyTorch Skip-Gram Word Embeddings script."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    corpus = [
        "the cat sits on the mat",
        "dogs and cats are animals",
        "the mat is soft and comfortable"
    ]

    window, dim, epochs = 2, 50, 100
    words = list(set(" ".join(corpus).lower().split()))
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for w, i in w2i.items()}

    pairs = []
    for s in corpus:
        s = s.lower().split()
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if i != j:
                    pairs.append((w2i[w], w2i[s[j]]))

    class SkipGram(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(words), dim)
            self.out = nn.Linear(dim, len(words))

        def forward(self, x):
            return self.out(self.emb(x))

    model = SkipGram()
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.01)

    for e in range(epochs):
        loss = 0
        for c, ctx in pairs:
            x, y = torch.tensor([c]), torch.tensor([ctx])
            opt.zero_grad()
            l = loss_fn(model(x), y)
            l.backward()
            opt.step()
            loss += l.item()
        if (e + 1) % 10 == 0:
            print(f"Epoch {e+1}, Loss: {loss:.4f}")

    emb = model.emb.weight.data
    with open("word_embeddings.txt", "w") as f:
        for i, v in enumerate(emb):
            f.write(i2w[i] + " " + " ".join(map(str, v.tolist())) + "\n")

    print("Training complete!")
    print("Embeddings saved to word_embeddings.txt")

    p = min(30, len(words) - 1)
    xy = TSNE(n_components=2, perplexity=p, random_state=0).fit_transform(emb.numpy())

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab20(range(len(words)))

    for i, w in enumerate(words):
        plt.scatter(xy[i, 0], xy[i, 1], color=colors[i], s=100)
        plt.annotate(w, (xy[i, 0], xy[i, 1]), fontsize=12)

    plt.title("t-SNE Word Embeddings")
    plt.grid()
    plt.show()


def program2():
    """Runs Keras Iris Dataset Classification ANN script."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelBinarizer
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input

    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelBinarizer()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        X_train, y_train, epochs=100, validation_split=0.2, batch_size=8, verbose=0
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def program3():
    """Runs CIFAR-10 CNN Image Classification script."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    history = model.fit(
        x_train, y_train, epochs=10, validation_split=0.2, batch_size=64
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def program4():
    """Runs MNIST Autoencoder Reconstruction script."""
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model

    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0

    inp = Input(shape=(784,))
    x = Dense(128, activation='relu')(inp)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(32, activation='relu')(x)

    x = Dense(64, activation='relu')(encoded)
    x = Dense(128, activation='relu')(x)
    decoded = Dense(784, activation='sigmoid')(x)

    autoencoder = Model(inp, decoded)
    encoder = Model(inp, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(
        x_train, x_train,
        epochs=50,
        batch_size=256,
        validation_data=(x_test, x_test),
        verbose=1
    )

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    plt.figure(figsize=(18, 6))
    for i in range(10):
        plt.subplot(3, 10, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis("off")
        plt.title("Original")

        plt.subplot(3, 10, i + 11)
        plt.imshow(encoded_imgs[i].reshape(8, 4), cmap='viridis')
        plt.axis("off")
        plt.title("Encoded")

        plt.subplot(3, 10, i + 21)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        plt.axis("off")
        plt.title("Reconstructed")

    plt.show()


def cli_entrypoint():
    """CLI runner allowing user to select a program to run."""
    print("Select a program to run:")
    print("1. Skip-Gram Word Embeddings (PyTorch)")
    print("2. Iris Dataset Classifier (Keras)")
    print("3. CIFAR-10 CNN (Keras)")
    print("4. MNIST Autoencoder (Keras)")
    choice = input("Enter choice (1-4): ").strip()
    if choice == "1":
        program1()
    elif choice == "2":
        program2()
    elif choice == "3":
        program3()
    elif choice == "4":
        program4()
    else:
        print("Invalid selection.")
