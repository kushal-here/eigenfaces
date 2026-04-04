import numpy as np
import cv2
import os
from PIL import Image

def load_faces(dataset_path, num_images=300):
    images = []
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            if len(images) >= num_images:
                return np.array(images, dtype=np.float32)
    return np.array(images, dtype=np.float32)

dataset_path = "lfw_funneled"
faces = load_faces(dataset_path)
print(f"Loaded {faces.shape[0]} images, each with {faces.shape[1]} pixels")

mean_face = np.mean(faces, axis=0)
centered = faces - mean_face

print("Running SVD...")
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
eigenfaces = Vt[:300].T
print(f"Eigenfaces shape: {eigenfaces.shape}")

# Find a person with at least 2 images
person_name = None
for p in os.listdir(dataset_path):
    p_path = os.path.join(dataset_path, p)
    if os.path.isdir(p_path) and len(os.listdir(p_path)) >= 2:
        person_name = p
        break

print(f"Morphing: {person_name}")
person_path = os.path.join(dataset_path, person_name)
imgs = os.listdir(person_path)

img1 = cv2.imread(os.path.join(person_path, imgs[0]), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(os.path.join(person_path, imgs[1]), cv2.IMREAD_GRAYSCALE)

img1 = cv2.resize(img1, (64, 64)).flatten().astype(np.float32)
img2 = cv2.resize(img2, (64, 64)).flatten().astype(np.float32)

face1 = img1
face2 = img2

def project(face, mean, eigenfaces):
    return eigenfaces.T @ (face - mean)

def reconstruct(coords, mean, eigenfaces):
    return eigenfaces @ coords + mean

coords1 = project(face1, mean_face, eigenfaces)
coords2 = project(face2, mean_face, eigenfaces)

print("Generating morph frames...")
frames = []
for i in range(60):
    t = i / 59
    interpolated = (1 - t) * coords1 + t * coords2
    reconstructed = reconstruct(interpolated, mean_face, eigenfaces)
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    frame = reconstructed.reshape(64, 64)
    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    frames.append(frame)

gif_frames = [Image.fromarray(f) for f in frames]
gif_frames[0].save("morph.gif", save_all=True, append_images=gif_frames[1:], duration=56, loop=0)
print("GIF saved as morph.gif")

# Save top 20 eigenfaces as a grid
print("Saving eigenface grid...")
n_show = 20
grid_cols = 5
grid_rows = 4
cell_size = 128

grid = np.zeros((grid_rows * cell_size, grid_cols * cell_size), dtype=np.uint8)

for i in range(n_show):
    ef = eigenfaces[:, i].reshape(64, 64)
    # Normalize to 0-255
    ef = ef - ef.min()
    ef = (ef / ef.max() * 255).astype(np.uint8)
    ef = cv2.resize(ef, (cell_size, cell_size))
    row = i // grid_cols
    col = i % grid_cols
    grid[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size] = ef

cv2.imwrite("eigenfaces_grid.png", grid)
print("Eigenface grid saved as eigenfaces_grid.png")