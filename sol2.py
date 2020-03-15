import numpy as np
from scipy.misc import imread
from scipy import signal
from skimage.color import rgb2gray
import scipy.io.wavfile
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt


"""======= Discrete Fourier Transform - DFT ======"""

def read_image(filename, representation):
    img = imread(filename)
    if representation == 1:
        img = rgb2gray(img)
        img = img.astype(np.float64)

    elif representation == 2:
        img = img.astype(np.float64)

    else:
        print("representation need to be '1' or '2'")
        return None

    return img / 255


"""
    This function transform a 1D discrete signal to its Fourier representation (use DFT)
    params: signal is an array of dtype float64 with shape (N,1)
    Returned value: complex Fourier signal and complex signal
"""
def DFT(signal):
    N = signal.shape[0]
    i = complex(0, 1)
    exp = np.exp((-2 * np.pi * i) / N)
    u = np.arange(N)
    first_col = np.power(exp, u)
    matrix = np.vander(first_col, increasing=True).T

    dft_matrix = np.dot(matrix, signal).astype(np.complex128)
    return dft_matrix


"""
    This function transform a Fourier representation to 1D discrete signal (use IDFT)
    params: fourier_signal is an array of dtype complex128 with the same shape
    Returned value: complex signal
"""
def IDFT(fourier_signal):
    N = fourier_signal.shape[0]
    i = complex(0, 1)
    exp = np.exp((2*np.pi*i)/N)
    u = np.arange(N)
    first_col = np.power(exp, u)
    matrix = np.vander(first_col, increasing=True).T
    matrix = matrix/N

    idft_matrix = np.dot(matrix, fourier_signal)
    return idft_matrix


"""
    This functions convert a 2D discrete signal to its Fourier representation
    params: image is a grayscale image of dtype float64
"""
def DFT2(image):
    height = image.shape[0]
    width = image.shape[1]
    row_dft = np.zeros((height, width), np.complex128)
    result = np.zeros((width, height), np.complex128)

    for row in range(height):
        signal = image[row]
        row_dft[row] = (DFT(signal))

    row_dft = row_dft.T

    for col in range(width):
        signal = row_dft[col]
        result[col] = (DFT(signal))

    result = result.T
    return result


"""
    This functions convert a Fourier representation to 2D discrete signal
    params: fourier_image e is a 2D array of dtype complex128
"""
def IDFT2(fourier_image):
    height = fourier_image.shape[0]
    width = fourier_image.shape[1]
    row_dft = np.zeros((height, width), np.complex128)
    result = np.zeros((width, height), np.complex128)

    for row in range(height):
        signal = fourier_image[row]
        row_dft[row] = (IDFT(signal))

    row_dft = row_dft.T

    for col in range(width):
        signal = row_dft[col]
        result[col] = (IDFT(signal))

    result = result.T
    return result


# Fs = 8000
# f = 5
# sample = 1000
# x = np.arange(sample)
# y = np.sin(2 * np.pi * f * x / Fs)
#
# dft = DFT(y)
# fft = np.fft.fft(y)
# print(np.isclose(dft, fft))
#
# idft = IDFT(dft)
# ifft = np.fft.ifft(dft)
# print(np.isclose(idft, ifft))
#
# filename = 'external/monkey.jpg'
# im = read_image(filename, 1)
# dft_2d = DFT2(im)
# fft_2d = np.fft.fft2(im)
# print(np.isclose(dft_2d, fft_2d))
#
# idft_2d = IDFT2(dft_2d)
# ifft_2d = np.fft.ifft2(dft_2d)
# print(np.isclose(idft_2d, ifft_2d))


"""======= Speech Fast Forward ======"""

"""
    This function changes the duration of an audio file by keeping the same samples,
    but changing the sample rate written in the file header
    params: filename- is a string representing the path to a WAV file
            ratio   - is a positive float64 representing the duration change
"""
def change_rate(filename, ratio):
    rate, data = scipy.io.wavfile.read(filename)
    scipy.io.wavfile.write("change_rate.wav", int(ratio*rate), data)


"""
    change the number of samples by the given ratio
    params: data  - a 1D ndarray of dtype float64 or complex128(*) representing
                    the original sample points
            ratio - is a positive float64 representing the duration change (0.25 < ratio < 4)
    Returned value: 1D ndarray of the dtype of data representing the new sample points
"""
def resize(data, ratio):
    N = data.shape[0]
    dft = DFT(data)
    dft_center = np.fft.fftshift(dft)

    odd = False
    if N%2 != 0:
        odd = True

    if ratio >= 1:
        slice = np.floor((N - (N/ratio))/2)
        slice = int(slice)
        if odd:
            dft_center = dft_center[slice: -slice-1]
        else:
            dft_center = dft_center[slice: -slice]

    else:
        slice = int(((N/ratio) - N)/2)
        if odd:
            dft_center = np.pad(dft_center, (slice, slice-1), mode="constant", constant_values=(0, 0))
        else:
            dft_center = np.pad(dft_center, (slice, slice), mode="constant", constant_values=(0, 0))

    new_dft = np.fft.ifftshift(dft_center)
    new_data = IDFT(new_dft)

    return new_data


"""
    fast forward function that changes the duration of an audio file by
    reducing the number of samples using Fourier
    but changing the sample rate written in the file header
    params: filename- is a string representing the path to a WAV file
            ratio   - is a positive float64 representing the duration change
"""
def change_samples(filename, ratio):
    rate, data = scipy.io.wavfile.read(filename)
    new_data = resize(data, ratio)
    new_data = np.real(new_data).astype(data.dtype)
    scipy.io.wavfile.write("change_samples.wav", int(rate), new_data)
    return new_data


####################################################################
# ex2_helper:

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    n = int(spec.shape[1] / ratio)
    time_steps = np.arange(n) * ratio
    time_steps = time_steps[time_steps < spec.shape[1]]

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
####################################################################

"""
    This function speeds up a WAV file, without changing the pitch,
    using spectrogram scaling. This is done by computing the spectrogram,
    changing the number of spectrogram columns, and creating back the audio.
    params: data - 1D ndarray of dtype float64 representing the original sample points
            ratio- a positive float64 representing the rate change of the WAV file (0.25 < ratio < 4)
"""
def resize_spectrogram(data, ratio):
    spectogram = stft(data)
    result = []

    # changing the number of spectrogram columns
    for row in spectogram:
        new_data = resize(row, ratio)
        result.append(new_data)

    result = np.array(result)
    result = istft(result)
    result = result.astype(np.int16)
    return result


"""
    This function speedups a WAV file by phase vocoding its spectrogram.
    params: data -  1D ndarray of dtype float64 representing the original sample points
            ratio- a positive float64 representing the rate change of the WAV file (0.25 < ratio < 4)
"""
def resize_vocoder(data, ratio):
    spec = stft(data)
    warped_spec = phase_vocoder(spec, ratio)
    new_spec = istft(warped_spec)
    new_spec = new_spec.astype(np.int16)
    return new_spec


# audio_filename = 'external/aria_4kHz.wav'
# rate, audio_data = scipy.io.wavfile.read(audio_filename)
# rate = 1.5

# change_rate(audio_filename, rate)


# reduced_samples = change_samples(audio_filename, rate)
# if not np.all(reduced_samples.shape[0] == (audio_data.shape[0] / rate)):
#     print('the new samples shape should be: ', (audio_data.shape[0] / rate), 'but is:', reduced_samples.shape[0])
# else:
#     print("good")


# rate1 = 1.5
# rate, data = scipy.io.wavfile.read(audio_filename)
# reduced_samples = resize_spectrogram(data, rate1)
# new_shape = 4960
# if not np.all(reduced_samples.shape[0] == new_shape):
#     print("error")
# scipy.io.wavfile.write("resize_spec.wav", rate, reduced_samples)


# a = np.array([[1], [1], [1]])
# print(a.shape)

#
# ratio = 1.5
# rate, data = scipy.io.wavfile.read(audio_filename)
# result = resize_vocoder(data, ratio)
# scipy.io.wavfile.write("resize_vocoder.wav", rate, result)


"""======= Image derivatives ======"""


"""
    This function computes the magnitude of image derivatives
"""
def conv_der(im):
    conv_x = np.array([[0.5, 0, -0.5]])
    conv_y = conv_x.T

    dx = scipy.signal.convolve2d(im, conv_x, "same")
    dy = scipy.signal.convolve2d(im, conv_y, "same")

    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
    return magnitude


"""
    This function computes the magnitude of image derivatives using Fourier transform
"""
def fourier_der(im):
    dft = DFT2(im)
    dft_center = np.fft.fftshift(dft)

    rows = im.shape[0]
    cols = im.shape[1]
    var = (2 * np.pi * complex(0, 1)) / (rows*cols)

    center_fourier_x = np.arange(-rows/2, rows/2) * var
    center_fourier_y = np.arange(-cols/2, cols/2) * var

    mat_x = np.dot(dft_center, np.diag(center_fourier_y))
    mat_y = np.dot(np.diag(center_fourier_x), dft_center)

    dx = IDFT2(mat_x)
    dy = IDFT2(mat_y)
    magnitude = np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)
    return magnitude


# filename = 'external/monkey.jpg'
# image = read_image(filename, 1)
# magnitude1 = conv_der(image)
# if not np.all(magnitude1.shape == image.shape):
#     print('derivative magnitude shape should be :', image.shape, 'but is:', magnitude1.shape)
# else:
#     print(image.shape)
#     print(magnitude1.shape)


# filename = 'external/monkey.jpg'
# image = read_image(filename, 1)
# magnitude1 = fourier_der(image)
# if not np.all(magnitude1.shape == image.shape):
#     print('derivative magnitude shape should be :', image.shape, 'but is:', magnitude1.shape)
# else:
#     print(image.shape)
#     print(magnitude1.shape)
#
# plt.imshow(magnitude1, cmap="gray")
# plt.show()

# print(np.arange(-5, 5) * 2)
