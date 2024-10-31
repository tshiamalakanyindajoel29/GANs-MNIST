import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Charger les données MNIST
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train - 127.5) / 127.5  # Normalisation entre -1 et 1
X_train = np.expand_dims(X_train, axis=3)  # Ajouter la dimension pour le canal de couleur

# Paramètres du GAN
latent_dim = 100  # Taille du vecteur aléatoire en entrée du générateur
img_shape = (28, 28, 1)  # Dimension des images d'entrée

# Construire le modèle du générateur
def build_generator():
    model = tf.keras.Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation="tanh"))
    model.add(Reshape(img_shape))
    return model

# Construire le modèle du discriminateur
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))
    return model

# Compiler le modèle GAN
def compile_gan(generator, discriminator):
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    discriminator.trainable = False  # Geler le discriminateur pour entraîner le générateur

    gan_input = tf.keras.Input(shape=(latent_dim,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(loss="binary_crossentropy", optimizer="adam")
    return gan

# Entraîner le GAN
def train_gan(generator, discriminator, gan, epochs=10000, batch_size=64, sample_interval=1000):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Entraînement du discriminateur
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Entraînement du générateur
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

        # Afficher la progression
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Sauvegarder les échantillons générés à intervalles réguliers
        if epoch % sample_interval == 0:
            sample_images(generator, epoch)

# Générer et afficher des images
def sample_images(generator, epoch, n=5):
    noise = np.random.normal(0, 1, (n * n, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Renormalisation pour affichage

    fig, axs = plt.subplots(n, n)
    count = 0
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap="gray")
            axs[i, j].axis("off")
            count += 1
    plt.show()

# Initialiser et entraîner le GAN
generator = build_generator()
discriminator = build_discriminator()
gan = compile_gan(generator, discriminator)
train_gan(generator, discriminator, gan)
