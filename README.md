# Cleverlytics Audio Anonymization Algorithm

## Summary

The CleverLytics Audio Anonymization Algorithm is a sophisticated audio processing tool designed to transform audio files while preserving their core acoustic characteristics. The algorithm employs multiple advanced techniques to anonymize audio recordings, making them difficult to trace back to the original source. 

Key anonymization techniques include:
- Spectral envelope modification using McAdams coefficient
- MFCC (Mel-Frequency Cepstral Coefficients) hashing
- Pitch shifting
- Optional noise reduction
- Gain adjustment

The algorithm ensures privacy by systematically altering audio signals through mathematically complex transformations that obscure identifying vocal characteristics.

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Step-by-Step Installation

1. Clone the repository:
```bash
git clone https://github.com/Cleverlytics/voice-privacy.git
cd voice-privacy
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
pip list | grep -E "numpy|librosa|soundfile|noisereduce|tqdm|scipy"
```

## Quick Usage

Run the script with the following command:
```bash
python3 anonymization.py <input_directory> <output_directory>
```
Replace `<input_directory>` with the folder containing your audio files and `<output_directory>` with the folder where anonymized files will be saved.

## Detailed Definitions

### 1. Spectral Envelope
The spectral envelope represents the overall shape of the frequency spectrum of an audio signal. It captures the distribution of energy across different frequencies, essentially defining the timbre or tone color of a sound. In this algorithm, the spectral envelope is systematically modified to obscure original vocal characteristics.

### 2. MFCC (Mel-Frequency Cepstral Coefficients)
MFCCs are coefficients that represent the short-term power spectrum of sound. They mimic human auditory system perception by converting traditional frequency scales to the mel scale. In our algorithm, MFCCs are cryptographically hashed to further anonymize the audio signal.

### 3. Pitch
Pitch represents the perception of how high or low a sound seems. In audio processing, pitch is related to the fundamental frequency of a sound wave. The algorithm shifts pitch by a specified number of semitones to alter the original audio's recognizability.

### 4. Formants
Formants are concentrations of acoustic energy around particular frequencies in a sound's spectrum. They are crucial in determining vowel quality and individual vocal characteristics. The McAdams coefficient helps transform these formant frequencies.

### 5. Gain
Gain refers to the amplification of an audio signal, measured in decibels (dB). Increasing gain raises the overall volume of the audio. In this algorithm, gain is used to normalize or modify the signal's amplitude.

### 6. McAdams Coefficient
The McAdams coefficient is a mathematical transformation parameter that modifies frequency relationships. By applying a non-linear exponent to frequency values, it creates a sophisticated method of altering spectral characteristics while maintaining some original signal properties.

### 7. Hashing
Hashing is a process of converting input data into a fixed-size string of bytes, typically using cryptographic hash functions like SHA-256. In this algorithm, audio features are hashed to introduce additional anonymization and prevent feature recognition.


## Contact

For additional support, please contact amine.boussetta@um6p.ma
