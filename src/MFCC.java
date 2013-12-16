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
	private static final double fStart = 64;
	private static final int NUM_MEL_CHANNELS = 24;
	
	/**
	 * 
	 * @param channels
	 * @param sampleRate - sample rate in Hz
	 * @return
	 */
	public static double[][] computeMFCC(double[][][] channels, double sampleRate) {
		// For each channel
		double[][] mfccs = new double[channels.length][];
		for(int i=0; i < channels.length; i++) {
			mfccs[i] = computeMFCC(channels[i][0], sampleRate);
		}
		return mfccs;
	}
	
	/**
	 * 
	 * @param channelVals
	 * @param sampleRate - sample rate in Hz
	 * @return
	 */
	public static double[] computeMFCC(double[] channelVals, double sampleRate) {
		// length in frames of the fft
		int fftl = channelVals.length;
		double[] melBank = new double[NUM_MEL_CHANNELS];
		
		//System.out.println("MFCC got:");
		//System.out.println(Arrays.toString(channelVals));
		
		// FFT
		Complex[] fft;
		fft = fftransformer.transform(channelVals, 
			  						  TransformType.FORWARD);
		
		double fs = sampleRate;
		double fsHalf = fs / 2D;
		double melFStart = mel(fStart);
		double melHalfSampleRate = mel(fsHalf);
		
		// The index of each mel channel in the fft bank
		int[] cbins = new int[25];		
		
		for(int i=0; i < cbins.length; i++) {
			
			if(i == 0) {
				cbins[i] = (int) Math.round( (fStart / fs) * fftl);
			} else if(i == cbins.length - 1) {
				cbins[i] = fftl / 2;
				
			} else {
			
				double quotient = ( melHalfSampleRate - melFStart ) / 24D;
				double fci = melInverse( melFStart + (quotient * i) );
				
				cbins[i] = (int) Math.round( (fci / fs) * fftl );
				
			}			
		}
		
		// Now compute the mel filter outputs
		for(int j=0; j < melBank.length; j++) {
			int k = j + 1;
			
			double sum = 0;
			for(int i=cbins[k - 1]; i <= k; i++) {
				double fftVal = fft[i].abs();
				double numerator = i - cbins[k-1] + 1;
				double denominator = cbins[k] - cbins[k-1] + 1;
				double val = (numerator / denominator) * fftVal;
				sum += val;
			}
			for(int i=cbins[k] + 1; i <= k+1; i++) {
				double fftVal = fft[i].abs();
				double numerator = i - cbins[k];
				double denominator = cbins[k+1] - cbins[k] + 1;
				double val = (1 - (numerator/denominator)) * fftVal;
				sum += val;
			}
			
			// The mel filter output
			melBank[j] = limitedLn(sum);
		}
		
		// Compute the 13-order mel-frequency cepstral coefficients from filter
		double[] mfccs = new double[13];
		for(int i=0; i < mfccs.length; i++) {
			double sum = 0;
			double quotient = (Math.PI * i / 23);
			for(int j=0; j < melBank.length; j++) {
				sum += melBank[j] * Math.cos(quotient * (j - .5));
			}
			mfccs[i] = sum; 
		}
		
		//System.out.println("Returning:");
		//System.out.println(Arrays.toString(mfccs));
		
		return mfccs;
	}
	
	public static double limitedLn(double x) {
		return Math.max(Math.log(x), -50);
	}
	
	public static double mel(double x) {
		return 2595D * Math.log10(1D + (x / 700D));
	}
	
	public static double melInverse(double x) {
		x = (x / 2595D);
		return  (Math.pow(10, x) -1D) * 700D;
	}
	
}
