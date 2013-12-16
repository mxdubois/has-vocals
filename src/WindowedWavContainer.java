import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;


public abstract class WindowedWavContainer implements IDataContainer {
	
	protected WavFile mWavFile;
	protected int mTotalFramesRead;
	protected int mWindowSize;
	protected int mSlideSize;
	protected int mBufferNumFrames;
	protected double[][] mBuffer;
	protected int mNewDataOffset = 0;
	protected int mBufferIdx = 0;
	protected int mChunkNum = -1;
	protected int mMaxBufferIdx;
	protected int mNumChannels;
	protected WindowConfig mWindowConfig;
	protected long mSampleRate;

	public WindowedWavContainer(WavFile wavFile) {
		this(wavFile, DEFAULT_WINDOW_CONFIG);
	}
	
	public WindowedWavContainer(WavFile wavFile, WindowConfig config) {
		mWavFile = wavFile;
		
		mWindowConfig = config;
		
		// Determine window length in frames
		mSampleRate = mWavFile.getSampleRate();
   	 	mWindowSize = mWindowConfig.getWindow(mSampleRate);
   	 	mSlideSize = mWindowConfig.getShift(mSampleRate);
   	 	
   	 	if(mWindowSize < 0 || mSlideSize < 0) {
   	 		throw new IllegalArgumentException("The .wav file you supplied "
   	 									+ "uses an unsupported sample rate.");
   	 	}
   	 	
   	 	// We want a buffer size that is divisible by both window and slide
   	 	mBufferNumFrames = mWindowSize * mSlideSize;
   	 	mMaxBufferIdx = mBufferNumFrames - mWindowSize;
   	 	mNumChannels = mWavFile.getNumChannels();
	}
	
	public int getWindowSize() {
		return mWindowConfig.getWindow(mSampleRate);
	}
	
	public int getShiftSize() {
		return mWindowConfig.getShift(mSampleRate);
	}
	
	@Override
	public void open() throws Exception {
		// Allocate memory for our buffer now
		// Each row in the buffer matrix is a channel
		mBuffer = new double[mNumChannels][mBufferNumFrames];
	}
	
	@Override
	public void close() throws Exception {
		mBuffer = null;
		mWavFile.close();
	}

	@Override
	public abstract boolean hasNext();

	@Override 
	public abstract LabeledData next() throws Exception;
	
	protected boolean hasNextWindow() {
		boolean readWholeFile = mTotalFramesRead == mWavFile.getNumFrames();
		boolean processedBuffer = mMaxBufferIdx - mBufferIdx < mSlideSize;
		boolean noSlidesInFile = mWavFile.getFramesRemaining() < mSlideSize;
		
		//System.out.println("readWholeFile: " + readWholeFile);
		//System.out.println("processedBuffer: " + processedBuffer);
		//System.out.println("noSlidesInFile: " + noSlidesInFile );
		
		return !(processedBuffer && (readWholeFile || noSlidesInFile));
	}

	protected LabeledData nextWindow() throws IOException , WavFileException{
		if(mChunkNum < 0 || mBufferIdx >= mMaxBufferIdx) {
			nextChunk();
			preProcessBuffer();
			mBufferIdx = 0;
		}
		
		LabeledData next = processFrames(mBufferIdx, mWindowSize);
		//System.out.println("returning nextWindow");
		//System.out.println("new mBufferIdx: " + mBufferIdx);
		mBufferIdx += mSlideSize;
		return next;
	}
	
	protected abstract void preProcessBuffer();
	protected abstract LabeledData processFrames(int offset, int length);
	
	protected int getNumSpannedFrames() {
		return (mBufferNumFrames - mBufferIdx);
	}
	
	protected int getSpannedFramesIdx() {
		int numToRead = mBufferNumFrames - getNumSpannedFrames();
		return numToRead;
	}
	
	/**
	 * Adapted from http://www.labbookpages.co.uk/audio/javaWavFiles.html
	 * @throws WavFileException
	 * @throws IOException
	 */
	protected void nextChunk() throws WavFileException, IOException {
		int numToRead = mBufferNumFrames;
		mNewDataOffset = 0;
		
		// Handle frames that span this buffer and the next 
		// due to a slideSize < windowSize
		if(mChunkNum > 0) {
			
			int spannedFrames = getNumSpannedFrames();
			numToRead = getSpannedFramesIdx();
			mNewDataOffset = spannedFrames;
			
			// Copy frames we still need to front of buffer in each channel
			// (A tad inefficient, but much simpler)
			for(int i=0; i < mNumChannels; i++) {
				System.arraycopy(mBuffer[i], numToRead, mBuffer, 0, spannedFrames);
			}
		}
		
		// If total frames in wavFile was not evenly divisible by window,
		if(mWavFile.getFramesRemaining() < numToRead) {
			// pad the buffer with zeroes
			// TODO what if we divided this padding between start/end of file?
			numToRead = (int) mWavFile.getFramesRemaining();
			int padOffset = mNewDataOffset + numToRead;
			// Fill the rest with zeros in every channel
			for(int i=0; i < mNumChannels; i++) {
				Arrays.fill(mBuffer[i], padOffset, mBufferNumFrames, 0);
			}
			int endIdx = padOffset + (padOffset % mWindowSize);
			mMaxBufferIdx = endIdx - mWindowSize;
		}
		
		// Read frames until we've successfully filled the buffer 
		// or read all frames in the file.
		System.out.println("NumToRead:" + numToRead);
		int framesRead = 0;
		do {
			framesRead += mWavFile.readFrames(mBuffer, 
											  mNewDataOffset + framesRead, 
											  numToRead);
			System.out.println("Frames read: " + framesRead);
		} while(framesRead != 0 && framesRead < numToRead);
		mTotalFramesRead += framesRead;
		mChunkNum++;
	}

	
	
	
	//--------------------------------------------------------------------------
	// STATIC STUFF
	//--------------------------------------------------------------------------
	
	public static final WindowConfig DEFAULT_WINDOW_CONFIG = 
			new WindowConfig();
	public static final long MILLIS_PER_SEC = 1000;
	
	/**
	 * Config object for overlapping windows at different sample rates
	 * @author Michael DuBois
	 */
	public static class WindowConfig {
		
		public static int IDX_8kHZ = 0;
		public static int IDX_11kHZ = 1;
		public static int IDX_16kHZ= 2;
		public static int IDX_44kHZ= 3;
		
		public static int RATE_8kHZ = 8000;
		public static int RATE_11kHZ = 11000;
		public static int RATE_16kHZ= 16000;
		public static int RATE_44kHZ= 44100;
		
		private int[] windows = new int[] {200, 256, 400, 1323};
		private int[] shifts = new int[] {80, 110, 160, 441};
		private int[] fftLengths = new int[] {256, 256, 512, 2048};
	
		/**
		 * Construct a custom WindowConfig
		 * @param windows
		 * @param shifts
		 */
		public WindowConfig(int[] windows, int[] shifts, int fftLengths[]) 
		{
			System.arraycopy(windows, 0, this.windows, 0, IDX_16kHZ + 1);
			System.arraycopy(shifts, 0, this.shifts, 0, IDX_16kHZ + 1);
			System.arraycopy(fftLengths, 0, this.fftLengths, 0, IDX_16kHZ + 1);
		}
		
		/**
		 * Construct the default window config 
		 * as specified in ETSI ES 201 108 V1.1.3 (2003-09).
		 */
		public WindowConfig() {
		}
		
		public int getWindow(long sampleRate) {
			int idx = sampleRateToIdx(sampleRate);
			int window = idx;
			if(idx > 0) {
				window = windows[idx];
			}
			return window;
		}
		
		public int getShift(long sampleRate) {
			int idx = sampleRateToIdx(sampleRate);
			int shift = idx;
			if(idx > 0) {
				shift = shifts[idx];
			}
			return shift;
		}
		
		public int getFFTLength(long sampleRate) {
			int idx = sampleRateToIdx(sampleRate);
			int fftLength = idx;
			if(idx > 0) {
				fftLength = fftLengths[idx];
			}
			return fftLength;
		}
		
		private int sampleRateToIdx(long sampleRate) {
			int idx = -1;
			if(sampleRate == RATE_8kHZ) {
				idx = IDX_8kHZ;
			} else if(sampleRate == RATE_11kHZ) {
				idx = IDX_11kHZ;
			} else if(sampleRate == RATE_16kHZ) {
				idx = IDX_16kHZ;
			} else if(sampleRate == RATE_44kHZ) {
				idx = IDX_44kHZ;
			}
			return idx;
		}
	}

}
