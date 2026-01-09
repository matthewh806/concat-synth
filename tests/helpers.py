import numpy as np

def generate_white_noise(duration_s : float = 0.1, sample_rate : int = 44100) -> np.ndarray:
    '''
    Generates a numpy array of white noise
    '''
    n_samples = int(duration_s * sample_rate)
    return np.random.uniform(low=-1.0, high=1.0, size=(n_samples,))


def generate_sine_wave(freq: float, duration_s, amp: float, sample_rate: int = 44100) -> np.ndarray:
    '''
    Generates a numpy array for a sine wave
    '''

    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)