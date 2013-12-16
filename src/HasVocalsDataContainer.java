import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;

public class HasVocalsDataContainer extends WindowedWavContainer {

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

	HasVocalsDataContainer(WavFile wavFile, double label) {
		super(wavFile);
		mLabel = label;
		
		mPrev = new double[mNumChannels];
		mPrevDCOF = new double[mNumChannels];
		mQueue = new ArrayBlockingQueue<LabeledData>(QUEUE_BOUND);
	}

	@Override
	public void open() throws Exception {
		super.open();
		
		if(!hasNextWindow())
			throw new Exception("There was a problem opening container.");
		
		mPadData = processFrames(0, mWindowSize);
		mPadData.setIsFake(true);
		
		// fill all but one position in the queue with empty pad data
		for(int i=0; i < QUEUE_BOUND - 1; i++) {
			LabeledData padData = new LabeledData(mPadData);
			mQueue.add(padData);
		}
		
		LabeledData firstWindow = nextWindow();
		// Add the first window at the last position
		mQueue.add(firstWindow);
		
		// compute derivatives and poll away pads until real data reaches head
		for(int i=0; i < QUEUE_BOUND - 1; i++) {
			pollData();
		}
		System.out.println("HasVocalsDataContainer opened.");
	}
	
	@Override
	protected void preProcessBuffer() {
		// For each channel
		for(int i=0; i < mNumChannels; i++) {
			// For each new data entry
			for(int j=mNewDataOffset; j < mBuffer[i].length; j++) {
				
				// Capture temps
				double prevDCOF = mPrevDCOF[i];
				
				
				// get DC Offset-compensated value
				double currDCOF = offsetCompensation(mBuffer[i][j], 
													 mPrev[i], 
													 mPrevDCOF[i]);
				
				// To perform this on future buffers, we store the 
				// last frame we processed, before and after
				mPrev[i] = mBuffer[i][j];
				mPrevDCOF[i] = currDCOF;
				
				//mBuffer[i][j] = currDCOF;
				mBuffer[i][j] = preEmphasisFilter(currDCOF, prevDCOF);
			}
		}
	}

	@Override
	protected LabeledData processFrames(int offset, int length) {
		double[] features, labels;
	
		double[] logEnergies = new double[mNumChannels];
		for(int i=0; i < mNumChannels; i++) {
			logEnergies[i] = logEnergy(mBuffer[i], offset, length);
		}
		
		int paddedLength = mWindowConfig.getFFTLength(mSampleRate);
		double[][][] hammedChannels = 
				hammingWindow(mBuffer, offset, length, paddedLength);
		
		double[][] mfccsByChannel = 
				MFCC.computeMFCC(hammedChannels, mSampleRate);
		
		int numMfccs = mfccsByChannel[0].length;
		int featPerChannel = numMfccs + 1;
		int fLength = mfccsByChannel.length * featPerChannel;
		features = new double[fLength];
		for(int i=0; i < mNumChannels; i++) {
			System.out.println(Arrays.toString(mfccsByChannel[i]));
			
			int cOffset = i * featPerChannel;
			features[cOffset] = logEnergies[i];
			for(int j=0; j < numMfccs; j++) {
				features[cOffset + 1 + j] = mfccsByChannel[i][j];
			}
		}
		
		labels = new double[]{ mLabel };
		return new LabeledData(features, labels);
	}
	
	@Override
	public boolean hasNext()  {
		return (mQueue.peek() != null && !mQueue.peek().isFake());
	}
	
	@Override
	public LabeledData next() throws Exception {
		return pollData();
	}
	
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
	 * call this method twice to get second derivatives.
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
							if(nextFeats.length <= j || prevFeats.length <= j) {
								cannotCompute = true;
								break;
							}
								
							numerator +=  k * (nextFeats[j] - prevFeats[j]);
							denominator += ( k * k );
						}
						if(cannotCompute)
							break;
						double delta = numerator / (2D * denominator);
						
						//System.out.println("q: " + q + ",i: " + i + ",j: " + j + ", delta: " + delta);
						
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
	 * Implemented as defined in ETSI ES 201 108 V1.1.3 (2003-09).
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
			
			if(c == 0) {
				//System.out.println("got");
				//System.arraycopy(buffer[c], offset, window[c][0], 0, length);
				//System.out.println(Arrays.toString(window[c][0]));
			}
			
			// For each value
			for(int n=0; n < N ; n++) {
				int i = n + offset;
				double coeff = .54D - .46D*( Math.cos(2D*Math.PI*n / (N-1D)) );
				window[c][0][n] = coeff * buffer[c][i];
			}
			if(c == 0) {
				//System.out.println("returned");
				//System.out.println(Arrays.toString(window[c][0]));
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
		//System.out.println(val + " ==> " + of);
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
		//System.out.println(val + " ==> " + filtered);
		return filtered;
	}

	

}
