import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;

/**
 * A container that preprocesses wav data into windowed LabeledData fit for
 * speech recognition.
 * 
 * The following preprocessing is applied:
 * * Removes DC offset
 * * Applies a pre-emphasis filter
 * * Computes the log-energy of the signal (first features)
 * * Computes the MFCC for the signal
 * * Computes the first & second derivative of features over DERIV_T windows
 * 
 * @author DuBious
 *
 */
public class SpeechDataContainer extends WindowedWavContainer {

	private double mLabel;
	protected double[] mPrev;
	protected double[] mPrevDCOF; // DC Offset-free
	
	// Number of elements we need in each direction to compute derivative
	public static final int DERIV_T = 2;
	// This needs to be big enough for us to compute second derivatives.
	public static final int QUEUE_BOUND = 3*DERIV_T + 1;
	
	public static final double PREEMPHASIS_COEFF = .97;
	
	// We don't really need the concurrency... but whatever
	protected ArrayBlockingQueue<LabeledData> mQueue;
	private LabeledData mPadData;

	SpeechDataContainer(File file, double label) {
		super(file);
		mLabel = label;

		mQueue = new ArrayBlockingQueue<LabeledData>(QUEUE_BOUND);
	}

	@Override
	public void open() throws DataUnavailableException {
		super.open();
		
		mPrev = new double[mNumChannels];
		mPrevDCOF = new double[mNumChannels];
		
		if(!hasNextWindow())
			throw new DataUnavailableException(
					"There was a problem opening container.");
		
		mPadData = processFrames(0, mWindowSize);
		mPadData.setIsFake(true);
		
		// fill all but one position in the queue with empty pad data
		for(int i=0; i < QUEUE_BOUND - 1; i++) {
			LabeledData padData = new LabeledData(mPadData);
			mQueue.add(padData);
		}
		
		LabeledData firstWindow;
		try {
			firstWindow = nextWindow();
			// Add the first window at the last position
			mQueue.add(firstWindow);
			
			// compute derivatives and poll away pads until real data reaches head
			for(int i=0; i < QUEUE_BOUND - 1; i++) {
				pollData();
			}
		} catch (IOException e) {
			throw new DataUnavailableException(e.getMessage());
		} catch (WavFileException e) {
			throw new DataUnavailableException(e.getMessage());
		}
		
	}
	
	@Override
	public void close() throws DataUnavailableException {
		super.close();
		mPrev = null;
		mPrevDCOF = null;
		mQueue = null;
		mPadData = null;
	}
	
	@Override
	protected void preProcessBuffer() {
		// For each channel
		for(int i=0; i < mNumChannels; i++) {
			// For each new data entry
			for(int j=mNewDataOffset; j < mLongBuffer[i].length; j++) {
				
				// Capture temps
				double prevDCOF = mPrevDCOF[i];
				
				
				// get DC Offset-compensated value
				double currDCOF = offsetCompensation(mLongBuffer[i][j], 
													 mPrev[i], 
													 mPrevDCOF[i]);
				
				// To perform this on future buffers, we store the 
				// last frame we processed, before and after
				mPrev[i] = mLongBuffer[i][j];
				mPrevDCOF[i] = currDCOF;
				
				//mBuffer[i][j] = currDCOF;
				mBuffer[i][j] = preEmphasisFilter(currDCOF, prevDCOF);
			}
		}
	}

	@Override
	protected LabeledData processFrames(int offset, int length) {
		long start = System.currentTimeMillis();
		double[] features, labels;
	
		double[] logEnergies = new double[mNumChannels];
		for(int i=0; i < mNumChannels; i++) {
			logEnergies[i] = logEnergy(mBuffer[i], offset, length);
		}
		
		int paddedLength = mWindowConfig.getFFTLength(mSampleRate);
		double[][][] hammedChannels = 
				hammingWindow(mBuffer, offset, length, paddedLength);
		
		double[][] mfccsByChannel = 
				Mfcc.computeMFCC(hammedChannels, mSampleRate, 13, 1);
		
		int numMfccs = mfccsByChannel[0].length;
		int featPerChannel = numMfccs + 1;
		int fLength = mfccsByChannel.length * featPerChannel;
		features = new double[fLength];
		for(int i=0; i < mNumChannels; i++) {
			//if(i == 0)
				//System.out.println(Arrays.toString(mfccsByChannel[i]));
			
			int cOffset = i * featPerChannel;
			features[cOffset] = logEnergies[i];
			for(int j=0; j < numMfccs; j++) {
				features[cOffset + 1 + j] = mfccsByChannel[i][j];
			}
		}
		long elapsed = System.currentTimeMillis() - start;
		//System.out.println("processing one window takes " + elapsed + "ms");
		labels = new double[]{ mLabel };
		return new LabeledData(features, labels);
	}
	
	@Override
	public boolean hasNext()  {
		return (mQueue.peek() != null && !mQueue.peek().isFake());
	}
	
	@Override
	public LabeledData next() throws DataUnavailableException {
		LabeledData datum;
		try {
			datum = pollData();
		} catch (IOException e) {
			throw new DataUnavailableException(e.getMessage());
		} catch (WavFileException e) {
			throw new DataUnavailableException(e.getMessage());
		}
		return datum;
	}
	
	/**
	 * Polls the derivatives queue for the next LabeledData
	 * @return
	 * @throws IOException
	 * @throws WavFileException
	 */
	private LabeledData pollData() throws IOException, WavFileException {
		LabeledData[] array = new LabeledData[mQueue.size()];
		appendDerivatives(mQueue.toArray(array), 2);
		LabeledData ret = mQueue.poll();
		if(hasNextWindow()) {
			mQueue.offer(nextWindow());
		} else {
			LabeledData padData = new LabeledData(mPadData);
			mQueue.offer(padData);
			//System.out.println("hasNextWindow returned false.");
			//System.out.println("totalFramesRead: " + mTotalFramesRead);
		}
		return ret;
	}
	
	/**
	 * Appends first derivatives within range (DERIV_T, length - DERIV_T]
	 * Appends second derivatives within range (DERIV_T, length - 2*DERIV_T + 1]
	 * If first derivatives have not already been appended, you will need to
	 * @param data
	 */
	private void appendDerivatives(LabeledData[] data, int order) {
		if(data.length <=0)
			return;
		
		// For each derivative order
		for(int q=1; q <= order; q++) {
			// Loop through all eligible locations in the array
			for(int i=DERIV_T; i < data.length - q*DERIV_T; i++) {
				LabeledData datum = data[i];
				int featLength = datum.baseFeatureLength();
				double[] feats = datum.getFeatures();
				double[] newFeats;
				// We can only take the second derivative in this range (]
				int min2 = DERIV_T;
				int max2 = data.length - q*DERIV_T + 1;

				if(datum.highestDerivative() == q - 1) {
					
					// Take derivative of elements starting at 
					// order-1 feature lengths
					// For first, this is zero, for second it's one featLength
					int offset = (q-1) * featLength;
					// Need a bigger array to hold the new derivatives
					newFeats= new double[featLength*(q+1)];
					System.arraycopy(feats, 0, newFeats, 0, featLength*q);
					
					// Compute derivative on featLength items from offset
					// This can be used to compute derivatives of derivatives
					// with the proper offset
					for(int j=offset; j < offset + featLength; j++) {
						double numerator =  0;
						double denominator = 0;
						boolean cannotCompute = false;
						for(int k=1; k <= DERIV_T; k++) {
							double[] prevFeats = data[i - k].getFeatures();
							double[] nextFeats = data[i + k].getFeatures();
							
							// If the next datum does not have the required
							// derivatives, we cannot do this computation
							if(nextFeats.length <= j || prevFeats.length <= j) {
								cannotCompute = true;
								break;
							}
								
							numerator +=  k * (nextFeats[j] - prevFeats[j]);
							denominator += ( k * k );
						}
						
						// If we were unable to do this computation, move on
						if(cannotCompute)
							break;
						
						// Else, compute the derivative
						double delta = numerator / (2D * denominator);
						
						// Store derivative of item at specified location
						newFeats[j + featLength] = delta;
					}
					
					datum.setFeatures(newFeats, featLength,  q);
				}
			}
		}
		
	}
	
	/**
	 * Returns an array containing the hamming window of length n.
	 * The returned array contains real and complex arrays for each channel
	 * each channel's array is of the correct dimension to be transformed 
	 * by the apache commons FFT.
	 * 
	 * Hamming implemented as defined in ETSI ES 201 108 V1.1.3 (2003-09).
	 *
	 * @return
	 */
	public static double[][][] hammingWindow(double[][] buffer, 
											 int offset, 
											 int length,
											 int paddedLength)
	{
		double[][][] window = new double[buffer.length][][];
		int N = length;
		// For each channel
		for(int c=0; c < buffer.length; c++) {
			window[c] = new double[2][];
			window[c][0] = new double[paddedLength];
			window[c][1] = new double[paddedLength];
			
			// For each value
			for(int n=0; n < N ; n++) {
				int i = n + offset;
				double coeff = .54D - .46D*( Math.cos(2D*Math.PI*n / (N-1D)) );
				window[c][0][n] = coeff * buffer[c][i];
			}
		}
		return window;
	}
	
	/**
	 * The logarithmic frame energy measure
	 * As defined in ETSI ES 201 108 V1.1.3 (2003-09).
	 * @param val
	 * @param lastVal
	 * @param lastValOF
	 * @return
	 */
	public static double logEnergy(double[] frames, int offset, int length) {
		double sum = 0;
		
		// Sum energy from offset to offset + length
		for(int i=offset; i < length; i++){ 
			int j = i - offset;
			sum += frames[i] * j * j;
		}
		
		if(sum <= 0)
			return 0;
		else 
			return Math.max(Math.log(sum), -50D);
	}
	
	/**
	 * Compensates for a constant background noise (like DC).
	 * Implemented as defined in ETSI ES 201 108 V1.1.3 (2003-09).
	 * @param val
	 * @param lastVal
	 * @param lastValOF
	 * @return
	 */
	public static double offsetCompensation(double val, 
										  double lastVal, 
										  double lastValOF) 
	{
		double of = val - lastVal + .999D*lastValOF;
		return of;
	}
	
	/**
	 * Improves the signal-to-noise ratio.
	 * Implemented as defined in ETSI ES 201 108 V1.1.3 (2003-09).
	 * @param val
	 * @param lastVal
	 * @return
	 */
	public static double preEmphasisFilter(double val, double lastVal) {
		double filtered = val - PREEMPHASIS_COEFF*lastVal;
		return filtered;
	}

	

}
