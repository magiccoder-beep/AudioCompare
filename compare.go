package main

import (
	"encoding/binary"
	"log"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"

	"github.com/mjibson/go-dsp/fft"
	"gonum.org/v1/gonum/floats"
)

// LoadPCM loads a PCM file and returns the audio data and sample rate
func LoadPCM(filePath string, sampleRate int) ([]int16, int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, 0, err
	}

	data := make([]byte, fileInfo.Size())
	_, err = file.Read(data)
	if err != nil {
		return nil, 0, err
	}

	audioData := make([]int16, len(data)/2)
	for i := range audioData {
		audioData[i] = int16(binary.LittleEndian.Uint16(data[i*2 : (i+1)*2]))
	}

	return audioData, sampleRate, nil
}

// DownsampleAudio downsamples the audio data
func DownsampleAudio(audioData []float64, originalRate, targetRate int) ([]float64, int) {
	factor := float64(originalRate) / float64(targetRate)
	downsampled := make([]float64, int(float64(len(audioData))/factor))
	for i := range downsampled {
		downsampled[i] = audioData[int(float64(i)*factor)]
	}
	return downsampled, targetRate
}

// NormalizeAudio normalizes the audio data
func NormalizeAudio(audioData []float64) []float64 {
	maxVal := floats.Max(audioData)
	for i := range audioData {
		audioData[i] /= maxVal
		if math.IsNaN(audioData[i]) || math.IsInf(audioData[i], 0) {
			audioData[i] = 0 // Set invalid values to 0
		}
	}
	return audioData
}

// LowPassFilter applies a low-pass filter to the audio data
func LowPassFilter(audioData []float64, cutoff, fs float64, order int) []float64 {
	nyquist := 0.5 * fs
	normalCutoff := cutoff / nyquist

	// Check if normalCutoff is within a valid range
	if normalCutoff <= 0 || normalCutoff >= 1 {
		log.Printf("Invalid cutoff frequency: normalCutoff = %f", normalCutoff)
		return audioData
	}

	b, a := butter(order, normalCutoff)

	// Validate filter coefficients
	for _, coeff := range append(b, a...) {
		if math.IsNaN(coeff) || math.IsInf(coeff, 0) {
			log.Printf("Invalid filter coefficients: b = %v, a = %v", b, a)
			return audioData
		}
	}

	return lfilter(b, a, audioData)
}

// Butter returns the coefficients for a Butterworth filter
func butter(order int, cutoff float64) ([]float64, []float64) {
	z, p, k := butterworth(order, cutoff)
	b, a := zp2tf(z, p, k)
	return b, a
}

// Butterworth filter design (poles, zeros, and gain)
func butterworth(order int, cutoff float64) ([]complex128, []complex128, float64) {
	var z []complex128
	var p []complex128
	for k := 0; k < order; k++ {
		theta := (float64(2*k+1) / float64(2*order)) * math.Pi
		p = append(p, cmplx.Exp(complex(0, theta))*complex(cutoff, 0))
	}
	k := math.Pow(cutoff, float64(order))
	return z, p, k
}

// Convert poles and zeros to transfer function coefficients
func zp2tf(z, p []complex128, k float64) ([]float64, []float64) {
	b := poly(z)
	a := poly(p)
	for i := range b {
		b[i] *= k
	}
	// Ensure valid coefficients
	for i := range b {
		if math.IsNaN(b[i]) || math.IsInf(b[i], 0) {
			b[i] = 0
		}
	}
	for i := range a {
		if math.IsNaN(a[i]) || math.IsInf(a[i], 0) {
			a[i] = 0
		}
	}
	return b, a
}

// Polynomial coefficients from roots
func poly(roots []complex128) []float64 {
	coeff := []float64{1}
	for _, r := range roots {
		coeff = convolve(coeff, []float64{-real(r), 1})
	}
	return coeff
}

// Convolution of two polynomials
func convolve(a, b []float64) []float64 {
	result := make([]float64, len(a)+len(b)-1)
	for i := range a {
		for j := range b {
			result[i+j] += a[i] * b[j]
		}
	}
	return result
}

// LFilter applies an IIR filter to the audio data
func lfilter(b, a, x []float64) []float64 {
	y := make([]float64, len(x))
	for i := range x {
		if math.IsNaN(x[i]) || math.IsInf(x[i], 0) {
			y[i] = 0
			continue
		}

		y[i] = b[0] * x[i]
		for j := 1; j < len(b); j++ {
			if i-j >= 0 {
				if math.IsNaN(x[i-j]) || math.IsInf(x[i-j], 0) {
					continue
				}
				y[i] += b[j] * x[i-j]
			}
		}
		for j := 1; j < len(a); j++ {
			if i-j >= 0 {
				if math.IsNaN(y[i-j]) || math.IsInf(y[i-j], 0) {
					continue
				}
				if math.IsNaN(a[j]) || math.IsInf(a[j], 0) {
					continue
				}
				if math.IsNaN(a[j]*y[i-j]) || math.IsInf(a[j]*y[i-j], 0) {
					continue
				}
				// y[i] -= a[j] * y[i-j]
			}
		}

		// Check for NaN or Inf values and set them to 0
		if math.IsNaN(y[i]) || math.IsInf(y[i], 0) {
			y[i] = 0
		}
	}
	return y
}

// ComputeFFT computes the FFT of the audio data
func ComputeFFT(audioData []float64, sampleRate int) ([]float64, []float64) {
	// Check for invalid values
	for i, val := range audioData {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			audioData[i] = 0 // Set invalid values to 0
		}
	}
	fftData := fft.FFTReal(audioData)
	freq := make([]float64, len(fftData))
	amplitude := make([]float64, len(fftData))
	for i, val := range fftData {
		freq[i] = float64(i) * float64(sampleRate) / float64(len(fftData))
		amplitude[i] = cmplx.Abs(val)
	}
	return freq, amplitude
}

func intsToFloats(ints []int16) []float64 {
	floats := make([]float64, len(ints))
	for i, v := range ints {
		floats[i] = float64(v)
	}
	return floats
}

// CalculateSimilarity calculates the similarity between two FFT results
func CalculateSimilarity(fft1, fft2 []float64) float64 {
	minLen := len(fft1)
	if len(fft2) < minLen {
		minLen = len(fft2)
	}
	fft1 = fft1[:minLen]
	fft2 = fft2[:minLen]
	similarity := floats.Dot(fft1, fft2) / (math.Sqrt(floats.Dot(fft1, fft1)) * math.Sqrt(floats.Dot(fft2, fft2)))
	if math.IsNaN(similarity) || math.IsInf(similarity, 0) {
		log.Printf("Invalid similarity value: fft1 = %v, fft2 = %v", fft1, fft2)
		return 0
	}
	return similarity
}

func main() {
	// Load the first PCM file
	audioData1, sampleRate1, err := LoadPCM("audio/audio-3771048239.pcm", 8000)
	if err != nil {
		log.Fatal(err)
	}

	// Downsample and normalize the first audio data
	targetRate := 4000
	audioData1Downsampled, _ := DownsampleAudio(intsToFloats(audioData1), sampleRate1, targetRate)
	log.Printf("Audio data after downsampling: %v", audioData1Downsampled[:10])
	audioData1Downsampled = NormalizeAudio(audioData1Downsampled)
	log.Printf("Audio data after normalization: %v", audioData1Downsampled[:10])
	audioData1Downsampled = LowPassFilter(audioData1Downsampled, 1500, float64(targetRate), 5)
	log.Printf("Audio data after low pass filter: %v", audioData1Downsampled[:10])
	_, fft1 := ComputeFFT(audioData1Downsampled, targetRate)
	log.Printf("FFT1 data: %v", fft1[:10])

	// Load and process blacklist files
	dirEntries, err := os.ReadDir("blacklist/")
	if err != nil {
		log.Fatal(err)
	}

	for _, entry := range dirEntries {
		if filepath.Ext(entry.Name()) == ".pcm" {
			blacklistAudioData, blacklistSampleRate, err := LoadPCM("blacklist/"+entry.Name(), 8000)
			if err != nil {
				log.Println(err)
				continue
			}

			blacklistAudioDataDownsampled, _ := DownsampleAudio(intsToFloats(blacklistAudioData), blacklistSampleRate, targetRate)
			log.Printf("Blacklist audio data after downsampling: %v", blacklistAudioDataDownsampled[:10])
			blacklistAudioDataDownsampled = NormalizeAudio(blacklistAudioDataDownsampled)
			log.Printf("Blacklist audio data after normalization: %v", blacklistAudioDataDownsampled[:10])
			blacklistAudioDataDownsampled = LowPassFilter(blacklistAudioDataDownsampled, 1500, float64(targetRate), 5)
			log.Printf("Blacklist audio data after low pass filter: %v", blacklistAudioDataDownsampled[:10])
			_, fft2 := ComputeFFT(blacklistAudioDataDownsampled, targetRate)
			log.Printf("FFT2 data: %v", fft2[:10])

			similarity := CalculateSimilarity(fft1, fft2)
			log.Printf("Similarity with %s: %f\n", entry.Name(), similarity)
		}
	}
}
