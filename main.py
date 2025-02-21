import numpy as np
import librosa
import cv2
import soundfile as sf
import matplotlib.pyplot as plt

# Parameters for STFT and spectrogram mapping
n_fft = 2400
hop_length = 1200
vmin = -80   # dB lower limit
vmax = 0     # dB upper limit

# Step 1: Load the input audio file using soundfile
y, sr = sf.read('input.wav')
# If stereo, convert to mono
if y.ndim > 1:
    y = np.mean(y, axis=1)

# Step 2: Compute the STFT and get the magnitude spectrogram
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
S_abs = np.abs(S)

# Step 3: Convert amplitude spectrogram to decibel (dB) scale
S_db = librosa.amplitude_to_db(S_abs, ref=np.max)
S_db_clipped = np.clip(S_db, vmin, vmax)

# Map the dB values linearly to the range [0, 255]
img = ((S_db_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

# Step 4: Save the spectrogram image as a JPEG file with compression
cv2.imwrite('spectrogram.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 10])

# Step 5: Read back the JPEG image
img_reconstructed = cv2.imread('spectrogram.jpg', cv2.IMREAD_GRAYSCALE)
# Convert pixel values back to dB values (inverse of the linear mapping)
S_db_reconstructed = img_reconstructed.astype(np.float32) / 255.0 * (vmax - vmin) + vmin
# Convert the dB spectrogram back to a linear amplitude spectrogram
S_reconstructed = librosa.db_to_amplitude(S_db_reconstructed)

# Step 6: Reconstruct the audio using the Griffin–Lim algorithm
y_reconstructed = librosa.griffinlim(S_reconstructed, n_iter=1, hop_length=hop_length, win_length=n_fft)

# Step 7: Amplify the reconstructed audio by a factor of 1.5
y_amplified = 50.0 * y_reconstructed
# Optionally clip the values to avoid clipping artifacts (assuming normalized audio in [-1, 1])
y_amplified = np.clip(y_amplified, -1.0, 1.0)

# Save the amplified audio to output.wav
sf.write('output.wav', y_amplified, sr)
