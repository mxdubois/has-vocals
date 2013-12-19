import java.util.Arrays;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;


public class MFCC {

	private static final FastFourierTransformer fftransformer = 
			new FastFourierTransformer(DftNormalization.STANDARD);
	
	// From ETSI ES 201 108 V1.1.3 (2003-09):
	// fStart = 64 Hz, roughly corresponds to the case where the full 
	// frequency band is divided into 24 channels
	private static final double fStart = 64D;
	private static final int NUM_MEL_CHANNELS = 24;
	
	/**
	 * 
	 * @param channels
	 * @param sampleRate - sample rate in Hz
	 * @return
	 */
	public static double[][] computeMFCC(double[][][] channels, double sampleRate, int numCoeff, int offsetCoeff) {
		// For each channel
		double[][] mfccs = new double[channels.length][];
		
		
		for(int i=0; i < channels.length; i++) {
			//System.out.println();
			//System.out.println("input:" + Arrays.toString(channels[i][0]));
			mfccs[i] = computeMFCC(channels[i][0], sampleRate, numCoeff, offsetCoeff);
			//System.out.println("mfccs:" + Arrays.toString(mfccs[i]));
		}
		return mfccs;
	}
	
	/**
	 * 
	 * @param channelVals
	 * @param sampleRate - sample rate in Hz
	 * @return
	 */
	public static double[] computeMFCC(double[] channelVals, double sampleRate, int numCoeff, int offsetCoeff) {
		// length in frames of the fft
		int fftl = channelVals.length;
		double[] melBank = new double[NUM_MEL_CHANNELS];
		
		//System.out.println("MFCC got:");
		//System.out.println(Arrays.toString(channelVals));
		
		// FFT
		Complex[] fft;
		fft = fftransformer.transform(channelVals,TransformType.FORWARD);
		
		//System.out.println("fft:" + Arrays.toString(fft));
		
		double fs = sampleRate;
		double fsHalf = fs / 2D;
		double melFStart = mel(fStart);
		double melFsHalf = mel(fsHalf);
		
		// The index of each mel channel in the fft bank
		int[] cbins = new int[25];		
		
		double quotient = ( melFsHalf - melFStart ) / (double) NUM_MEL_CHANNELS;
		for(int i=0; i < cbins.length; i++) {
			
			if(i == 0) {
				cbins[i] = (int) Math.round( (fStart / fs) * fftl);
			} else if(i == cbins.length - 1) {
				cbins[i] = fftl / 2;
				
			} else {
			
				double fci = melInverse( melFStart + (quotient * ((double)i)) );
				
				cbins[i] = (int) Math.round( (fci / fs) * ((double)fftl) );
				
			}			
		}
		
		// From ETSI ES 201 108 V1.1.3 (2003-09):
		// The output of the mel filter is the weighted sum of the 
		// FFT magnitude spectrum values in each band
		// The half-overlapped windowing is used as follows:
		for(int k=1; k < cbins.length - 1; k++) {
			double sum = 0D;
			
			// First portion of sum
			double denominator = cbins[k] - cbins[k-1] + 1;
			for(int i=cbins[k - 1]; i <= cbins[k]; i++) {
				double fftVal = fft[i].abs();
				double numerator = i - cbins[k-1] + 1;
				double val = (numerator / denominator) * fftVal;
				sum += val;
			}
			
			// Second portion of sum
			denominator = cbins[k+1] - cbins[k] + 1;
			for(int i=cbins[k] + 1; i <= cbins[k+1]; i++) {
				double fftVal = fft[i].abs();
				double numerator = i - cbins[k];
				double val = (1 - (numerator/denominator)) * fftVal;
				sum += val;
			}
			
			// The filter output is subjected to a limited logarithm function
			melBank[k - 1] = limitedLn(sum);
		}
		
		// Compute the 13-order mel-frequency cepstral coefficients from filter
		double[] mfccs = new double[numCoeff];
		double denominator = NUM_MEL_CHANNELS - 1;
		for(int i=offsetCoeff; i < mfccs.length; i++) {
			// Start calculating coeffs at the given coeff offset
			int j = i + offsetCoeff;
			
			double sum = 0;
			// Sum the Discrete Cosine Transforms
			quotient = (Math.PI * ((double)j) / denominator);
			for(int k=0; k < melBank.length; k++) {
				int n = k + 1;
				double d = melBank[k] * Math.cos(quotient * (n - .5D));
				sum += d;
			}
			
			mfccs[i] = sum; 
		}
		
		//System.out.println("Returning:");
		//System.out.println(Arrays.toString(mfccs));
		
		return mfccs;
	}
	
	public static double limitedLn(double x) {
		return Math.max(Math.log(x), -50D);
	}
	
	public static double mel(double x) {
		return 2595D * Math.log10(1D + (x / 700D));
	}
	
	public static double melInverse(double x) {
		return (Math.pow(10, x/2595D) -1D) * 700D;
	}
	
}
