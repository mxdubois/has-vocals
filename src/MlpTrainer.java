import java.io.PrintStream;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.math3.stat.regression.SimpleRegression;

import com.sun.xml.internal.ws.util.StringUtils;


public class MlpTrainer {


	
	// CLI Output formatting
	public static final int CONSOLE_WIDTH = 80;
	public static final char PROGRESS_CHAR = '=';
	public static final char PROGRESS_EDGE_CHAR = '|';
	public static final int EPOCH_CELL_SIZE = 4;
	public static final int ERROR_CELL_SIZE = 10;
	
	// Sleep time in ms between checks if threads are complete
	private static final long SLEEP_TIME_START = 2;
	private static final long SLEEP_TIME_INCREMENT = 10;
	private static final long SLEEP_TIME_MAX = 150;
	
	public static void main(String[] args) {
		
	}
	
	private Mlp mMainNet;
	Mlp.Layer[] mTaskLayers;
	private PrintStream mOut;
	private int mEpoch = 0;
	private double mLastError = Double.NaN;
	private double mLearningRate = 1;
	
	private TrainingTask[] mTrainingTasks;
	private List<Future<Mlp>> mTrainingTaskFutures;
	private ExecutorCompletionService<Mlp> mTrainingEcs;
	
	private TestingTask[] mTestingTasks;
	private List<Future<Double>> mTestingTaskFutures;
	private ExecutorCompletionService<Double> mTestingEcs;
	
	private String mLastErrorStr;
	private ExecutorService mExecutor;
	private int mNumTrainingContainers;
	private int mNumTestingContainers;
	private double mLastDeltaError;
	private String mLastDeltaErrorStr;

	/**
	 * Constructs an Mlp trainer.
	 * @param net
	 * @param out
	 */
	public MlpTrainer(Mlp net, PrintStream out) {
		mMainNet = net;
		mOut = out;
	}
	
	/**
	 * Pads a string with specified character
	 * @param character
	 * @param length
	 * @return
	 */
	// TODO DONT LET THIS CRASH THE PROGRAM!!
	public static String paddedString(char character, int length) {
		char[] array = new char[length];
	    Arrays.fill(array, character);
	    return new String(array);
	}
	
	/**
	 * Pads a string with spaces to the specified size
	 * @param value
	 * @param size
	 * @return
	 */
	public static String paddedCell(String value, int size) {
		return value + paddedString(' ', size - value.length());
	}
	
	/**
	 * Prints the status of training.
	 * @param progress
	 */
	public void updateTrainingStatus(double progress) {
		progress = Math.max(Math.min(progress,1), 0);
		String status = "E%:  ";
		status += paddedCell("" + mLastErrorStr, ERROR_CELL_SIZE);
		status += "| dE%:  ";
		status += paddedCell("" + mLastDeltaErrorStr, ERROR_CELL_SIZE);
		status += " | Epoch: ";
		status += paddedCell("" + mEpoch, EPOCH_CELL_SIZE);
		printWithProgress(status, progress);
	}
	
	/**
	 * Prints the status of testing.
	 * @param progress
	 */
	public void updateTestingStatus(double progress) {
		progress = Math.max(Math.min(progress,1), 0);
		String status = "TESTING... | Epoch: ";
		status += paddedCell("" + mEpoch, EPOCH_CELL_SIZE);
		printWithProgress(status, progress);
	}
	
	/**
	 * Prints a string with an appended progress bar.
	 * @param status
	 * @param progress
	 */
	public void printWithProgress(String status, double progress) {
		int progressBarLength = CONSOLE_WIDTH - 2 - status.length();
		int segments = (int) (progress * progressBarLength);
		// Left pad with PROGRESS_CHAR
		String progressBar = paddedString(PROGRESS_CHAR, segments);
		progressBar += paddedString(' ', (progressBarLength - segments));
	    status += PROGRESS_EDGE_CHAR + progressBar + PROGRESS_EDGE_CHAR;
		mOut.print(status + "\r");
	}
	
	/**
	 * Ensures that the network can handle data's feature vector length.
	 * @param trainingContainers
	 */
	public void prepareNetwork(IDataContainer[] trainingContainers) {
		try {
			trainingContainers[0].open();
			LabeledData datum = trainingContainers[0].next();
			if(datum != null){
				double[] outputs = 
					mMainNet.evaluate(datum.getFeatures(), true);
				mOut.println("Outputs: " + Arrays.toString(outputs));
			} else {
				throw new IllegalStateException(
						"First container wouldn't open. duhdur.");
			}
		} catch (IDataContainer.DataUnavailableException e) {
			e.printStackTrace();
		} finally {
			try { trainingContainers[0].close();} catch (Exception e) {}
		}
	}
	
	/**
	 * Trains the trainer's assigned mlp.
	 * @param trainingContainers
	 * @param testingContainers
	 * @param minDeltaError
	 * @param maxEpochs
	 * @param maxThreads
	 * @return
	 */
	public int trainMlp(IDataContainer[] trainingContainers,
							IDataContainer[] testingContainers, 
								   double minDeltaError,
								   int maxEpochs,
								   int maxThreads) 
	{
		// Setup
		trainingInit(trainingContainers, testingContainers, maxThreads);
		
		mOut.println("Beginning training session...");
		long startTime = System.currentTimeMillis();

		try {
			// compute the initial error for reference
			computeError(testingContainers);
			mLastDeltaError = 0;
			
			// While the network has not yet converged,
			boolean converged = false;
			while(!converged) {
				long epochStart = System.currentTimeMillis();
				updateTrainingStatus(0); // 0% progress
				
				// Update the learning rate with decay
				// TODO expose decay as a config var
				mLearningRate = 1D / (.01*mEpoch + 1D);
				
				mTrainingTaskFutures.clear();
				mTestingTaskFutures.clear();
				
				//Train
				train();
				
				// Accumulate weights to mMainNet and propagate back to threads
				adjustWeights();
				
				// Estimate error on testing dataset
				computeError(testingContainers);
				
				long elapsed = System.currentTimeMillis() - epochStart;
				mOut.println("Epoch " + mEpoch + " complete. " 
								+ elapsed +"ms.\n");
				
				// Check convergence conditions
				if(Math.abs(mLastDeltaError) < minDeltaError) {
					converged = true;
					mOut.println("|mLastDeltaError| < minDeltaError : |"+ mLastDeltaError 
								+"| < " + minDeltaError + "");
				} else if(mEpoch >= maxEpochs) {
					converged = true;
					mOut.println("WARNING: Exceeded max epochs.");
				}
				mEpoch++;
	
			} // end while
		} catch(InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		mExecutor.shutdownNow();
		long elapsed = System.currentTimeMillis() - startTime;
		mOut.println("--- Your Neural Network ---");
		mOut.println(mMainNet.toString());
		mOut.println("--- END Your Neural Network ---");
		mOut.println("Training complete. Congratulations! ("
						+ mEpoch + " epochs | " 
						+ elapsed + "ms)");
		
		return mEpoch;
	}
	
	/**
	 * Sets up the training stage
	 * @param trainingContainers
	 * @param testingContainers
	 * @param maxThreads
	 */
	private void trainingInit(IDataContainer[] trainingContainers,
							 IDataContainer[] testingContainers,
							 int maxThreads) {
		mEpoch = 0;
		mLastError = testingContainers.length;
		
		mNumTrainingContainers = trainingContainers.length;
		mNumTestingContainers = testingContainers.length;
		
		// Make sure the network is prepared for data's feature vector length
		prepareNetwork(trainingContainers);
		
		// Determine the correct number of threads
		int numThreads = Runtime.getRuntime().availableProcessors();
		numThreads = Math.min(numThreads, maxThreads);
		
		// testing set < training set, so may need less threads
		int numTrainingTasks = Math.min(numThreads, trainingContainers.length);
		int numTestingTasks = Math.min(numThreads, trainingContainers.length);
		
		// Arrays to keep track of the tasks
		mTrainingTasks = new TrainingTask[numTrainingTasks];
		mTestingTasks = new TestingTask[numTestingTasks];
		mTrainingTaskFutures = new ArrayList<Future<Mlp>>();
		mTestingTaskFutures = new ArrayList<Future<Double>>();
		
		// We'll need these to propagate changes in mlp to each task
		int totalTasks = mTrainingTasks.length + mTestingTasks.length;
		mTaskLayers = new Mlp.Layer[totalTasks];
		
		// this is the number of threads we need to allocate
		numThreads = Math.max(numTrainingTasks, numTestingTasks);
		mExecutor = Executors.newFixedThreadPool(numThreads);
		
		// Tell the user what we'll be using
		mOut.println("Using " + numThreads + " threads!");

		initTrainingCallables(trainingContainers);
		initTestingCallables(testingContainers);
	}
	
	private void initTrainingCallables(IDataContainer[] trainingContainers) {
		mTrainingEcs = new ExecutorCompletionService<Mlp>(mExecutor);
		// Determine how to split the data among threads
		int trainContainersPerThread = 
				trainingContainers.length / mTrainingTasks.length;
		
		mOut.println("trainContainersPerThread: " + trainContainersPerThread);
		
		// Create the training callables
		for(int i=0; i < mTrainingTasks.length; i++) {
			// thread's offset in data container list
			int cOffset = trainContainersPerThread * i;
			int cEndIdx = cOffset + trainContainersPerThread;
			
			// Get subset of data for thread
			IDataContainer[] subset =  
					new IDataContainer[trainContainersPerThread];
			List<IDataContainer> subsetList = Arrays.asList(trainingContainers);
			subsetList.subList(cOffset, cEndIdx).toArray(subset);
			
			// Create the task with it's own copy of the network
			Mlp threadNet = new Mlp(mMainNet);
			mTrainingTasks[i] = new TrainingTask(threadNet, subset);
		}
	}
	
	private void initTestingCallables(IDataContainer[] testingContainers) {
		mTestingEcs = new ExecutorCompletionService<Double>(mExecutor);
		int testContainersPerThread = 
				testingContainers.length / mTestingTasks.length;
		
		mOut.println("testContainersPerThread: " + testContainersPerThread);
		
		// Create the testing callables
		for(int i=0; i < mTestingTasks.length; i++) {
			// thread's offset in data container list
			int cOffset = testContainersPerThread * i;
			int cEndIdx = cOffset + testContainersPerThread;
			
			// Get subset of data for thread
			IDataContainer[] subset =  
					new IDataContainer[testContainersPerThread];
			List<IDataContainer> subsetList = Arrays.asList(testingContainers);
			subsetList.subList(cOffset, cEndIdx).toArray(subset);
			
			// Create the task with it's own copy of the network
			Mlp threadNet = new Mlp(mMainNet);
			mTestingTasks[i] = new TestingTask(threadNet, subset);
		}
	}
	
	private void adjustWeights()
			throws ExecutionException, InterruptedException 
	{
		mOut.println(" Compiling results...");
		long compilingStart = System.currentTimeMillis();
		
		Mlp.Layer current;
		current = mMainNet.getHead();
		
		// Reset threadLayers to head... and set phasers to stun!
		int numTrainingTasks = mTrainingTaskFutures.size();
		int numTestingTasks = mTrainingTaskFutures.size();
		for(int i=0; i < numTrainingTasks; i++) {
			Mlp mlp = mTrainingTaskFutures.get(i).get();
			mTaskLayers[i] = mlp.getHead();
		}
		for(int i=0; i < mTestingTasks.length; i++) {
			Mlp mlp = mTestingTasks[i].getMlp();
			mTaskLayers[numTrainingTasks + i] = mlp.getHead();
		}
		
		// Adjust weights from head to tail and copy to all threads
		while(current != null) {
			
			// For each node in this layer
			for(int i=0; i < current.size(); i++) {
				
				double[] deltaWeights = current.getDeltaWeightsAt(i);
				
				// Accumulate deltaWeights from TrainingTasks
				// (but not from TestingTasks!!)
				for(int j=0; j < numTrainingTasks; j++) {
					double[] threadDeltaWeights = 
							mTaskLayers[j].getDeltaWeightsAt(i);
					Vector.scale(threadDeltaWeights, 1D /mTrainingTaskFutures.size() );
					Vector.addTo(threadDeltaWeights, deltaWeights);
					
					// Reset deltaWeights for this node in the task
					Arrays.fill(threadDeltaWeights, 0);
				}
				
				
				// Scale deltaWeights in-place
				Vector.scale(deltaWeights, mLearningRate);
				double[] weights = current.getWeightsAt(i);
				// Adjust weights in-place
				Vector.addTo(deltaWeights, weights);
				
				// Copy weights for this node to both Training and TestingTasks
				for(int j=0; j < mTaskLayers.length; j++) {
					double[] threadWeights = mTaskLayers[j].getWeightsAt(i);
					System.arraycopy(weights, 0, 
									 threadWeights, 0, 
									 weights.length);
				}
				
				// Reset deltaWeights for this node in main
				Arrays.fill(deltaWeights, 0);
			}
			
			// Advance all layers
			current = current.next();
			for(int i=0; i < mTaskLayers.length; i++) {
				mTaskLayers[i] = mTaskLayers[i].next();
			}
		}
		
		long elapsed = System.currentTimeMillis() - compilingStart;
		mOut.println("Finished compiling results." 
						+ elapsed + "ms.");
	}

	/**
	 * Runs one iteration of training with TrainingTask callables.
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */
	private void train() throws InterruptedException, ExecutionException {
		long trainingStart = System.currentTimeMillis();
		
		// Submit threads to executor
		for(int i=0; i < mTrainingTasks.length; i++) {
			mTrainingTasks[i].onSubmit();
			mTrainingEcs.submit(mTrainingTasks[i]);
		}
		
		// Wait for them all to finish
		long sleepTime = SLEEP_TIME_START;
		while(mTrainingTaskFutures.size() < mTrainingTasks.length) {
			Future<Mlp> future = 
					mTrainingEcs.poll(1, TimeUnit.NANOSECONDS);
			
			if(future != null) {
				sleepTime = SLEEP_TIME_START;
				mTrainingTaskFutures.add(future);
			} else {
				sleepTime += SLEEP_TIME_INCREMENT;
				sleepTime = Math.min(sleepTime, SLEEP_TIME_MAX);
			}
			
			// Update the progress bar
			double processed = 0;
			for(TrainingTask task : mTrainingTasks) {
				processed += task.getNumProcessed();
			}
			updateTrainingStatus(processed / mNumTrainingContainers);
			Thread.sleep(sleepTime);
		}
		
		mOut.println(); // to clear status line
		long elapsed = System.currentTimeMillis() - trainingStart;
		mOut.println("Epoch Training complete. " + elapsed + "ms.");
	}
	
	/**
	 * Runs testing to compute the error, updates error vars
	 * @param testingContainers
	 * @return the change in error from last computation
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */
	private void computeError(IDataContainer[] testingContainers) 
			throws InterruptedException, ExecutionException 
	{
		double newError;
		mOut.println("Computing error...");
		long testingStart = System.currentTimeMillis();
		
		// Compute delta Error
		newError = test();
		mLastDeltaError = newError - mLastError;
		
		// Update error tracking vars
		mLastError = newError;
		double lastErrorPercent = mLastError * 100D;
		double lastDeltaErrorPercent = mLastDeltaError * 100D;
		mLastErrorStr = "" 
				+ (Math.round(lastErrorPercent * 10000D) / 10000D);
		mLastDeltaErrorStr = "" 
				+ (Math.round(lastDeltaErrorPercent * 10000D) / 10000D);
		
		long elapsed = System.currentTimeMillis() - testingStart;
		mOut.println("Error Computed (" + mLastError + "%)." 
				+ elapsed + "ms.");
	}
	
	/**
	 * Runs one iteration of testing with TestingTask callables.
	 * @return
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */
	private double test() throws InterruptedException, ExecutionException 
	{
		// Submit testing threads to executor
		for(int i=0; i < mTestingTasks.length; i++) {
			mTestingTasks[i].onSubmit();
			mTestingEcs.submit(mTestingTasks[i]);
		}
		
		// Wait for them all to finish
		long sleepTime = SLEEP_TIME_START;
		while(mTestingTaskFutures.size() < mTestingTasks.length) {
			Future<Double> future = 
					mTestingEcs.poll(1, TimeUnit.NANOSECONDS);
			
			if(future != null) {
				sleepTime = SLEEP_TIME_START;
				mTestingTaskFutures.add(future);
			} else {
				sleepTime += SLEEP_TIME_INCREMENT;
				sleepTime = Math.min(sleepTime, SLEEP_TIME_MAX);
			}
			
			// Update the progress bar
			double processed = 0;
			for(TestingTask task : mTestingTasks) {
				processed += task.getNumProcessed();
			}
			updateTestingStatus(processed / mNumTestingContainers);
			Thread.sleep(sleepTime);
		}
		
		double n = 0;
		double sum = 0;
		// Reset threadLayers to head... and set phasers to stun!
		for(int i=0; i < mTestingTaskFutures.size(); i++) {
			Double result = mTestingTaskFutures.get(i).get();
			if(result != null) {
				sum += result.doubleValue();
				n++;
			}
		}
		
		mOut.println();
		
		// Return the average
		return sum / n;
	}
	
	/**
	 * A Callable that returns the mean square error of this task's slave
	 * network on this task's assigned testing subset. For proper results, 
	 * one must update this task's slave network between each run.
	 * @param testingContainers
	 * @return
	 */
	private class TestingTask implements Callable<Double> {
		Mlp mMlp;
		IDataContainer[] mTestingContainers;
		AtomicInteger numContainersProcessed;
		
		/**
		 * Constructs a TestingTask.
		 * @param mlp - in most cases, a deep copy of a main Mlp to test
		 * @param subset - the data subset to test on
		 */
		TestingTask(Mlp mlp, IDataContainer[] subset) {
			mMlp = mlp;
			mTestingContainers = subset;
			numContainersProcessed = new AtomicInteger(0);
		}
		public Mlp getMlp() {
			return mMlp;
		}

		public double getNumProcessed() {
			return numContainersProcessed.get();
		}
		
		public void onSubmit() {
			numContainersProcessed.set(0);
		}

		@Override
		public Double call() throws Exception {
			numContainersProcessed.set(0);
			double sumSqrResiduals = 0;
			int trials = 0;
			int targetsLength = 0;
			
			for(IDataContainer dataContainer : mTestingContainers) {
			
				LabeledData datum;
				try {
					dataContainer.open();
					
					while(dataContainer.hasNext()) {
						
							datum = dataContainer.next();
							
							double[] targets = datum.getLabels();
							double[] feats = datum.getFeatures();
							double[] outputs = mMainNet.evaluate(feats, true);
							
							targetsLength = targets.length;
							
							double[] diff = Vector.sub(targets, outputs);
							//mOut.println("diff: " + Arrays.toString(diff));
							sumSqrResiduals += Vector.dot(diff, diff);
							//mOut.println(sumSqrResiduals);
							trials++;
						
					}
				} catch(Exception e) {
					e.printStackTrace();
				} finally {
					try { dataContainer.close(); } catch(Exception e) {}
				}
				
				numContainersProcessed.addAndGet(1);
			}
			double quantity = trials * targetsLength;
			
			return new Double(sumSqrResiduals / quantity);
		}
	} // End TestingTask
	
	/**
	 * A Callable that returns an Mlp with deltaWeights trained on this task's
	 * assigned training subset. For proper results, one must update this
	 * task's slave network between each run.
	 * @author DuBious
	 *
	 */
	private class TrainingTask implements Callable<Mlp> {

		Mlp mMlp;
		IDataContainer[] mDataContainers;
		AtomicInteger numContainersProcessed;
		int numDataProcessed = 0;
		
		/**
		 * Constructs a TrainingTask.
		 * @param mlp - in most cases, a deep copy of a main Mlp to train.
		 * @param subset
		 */
		TrainingTask(Mlp mlp, IDataContainer[] subset) {
			mMlp = mlp;
			mDataContainers = subset;
			numContainersProcessed = new AtomicInteger(0);
		}
		
		/**
		 * (thread-safe) Returns the number of data containers processed 
		 * @return
		 */
		public int getNumProcessed() {
			return numContainersProcessed.get();
		}
		
		public void onSubmit() {
			numContainersProcessed.set(0);
		}
		
		@Override
		public Mlp call() throws Exception {
			numContainersProcessed.set(0);
			numDataProcessed = 0;
			for(IDataContainer dataContainer : mDataContainers){
				LabeledData datum;
				try {
					dataContainer.open();
					while(dataContainer.hasNext()) {
						
							datum = dataContainer.next();
						
							double[] outputs = 
									mMlp.evaluate(datum.getFeatures(), true);
							double[] targets = datum.getLabels();
							
		
							numDataProcessed++;
							computeBlames(outputs, targets);
							updateDeltaWeights();
							
						
					}
					numContainersProcessed.addAndGet(1);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} finally {
					try { dataContainer.close(); } catch(Exception e) {}
				}
				
			}
			
			// Compute the average deltaWeight across all cases
			averageDeltaWeights();
			
			return mMlp;
		}
		
		/**
		 * Computes and assigns a blame factor vector for each layer
		 * In academic papers this is often denoted with a lowercase delta
		 * TODO don't assume output is the only layer with activation function
		 * @param outputs
		 * @param targets
		 */
		private void computeBlames(double[] outputs, double[] targets) {
			// For output nodes, the blame is just assumed to be the 
			// difference between outputs and targets
			double[] nextBlames = Vector.sub(targets, outputs);
			double[] currentBlames;
			
			int layerIdx = mMlp.size() - 1;
			Mlp.Layer current = mMlp.getTail();

			// Get the output layer's activation function
			Mlp.IActivationFunction act = 
					current.getActivationFunction();
			
			// Back-propagate from tail
			while(current != null) {
				currentBlames = current.getBlames();
				
				// For each node in this layer
				for(int i=0; i < outputs.length; i++) {
					
					
					// The errorContrib is a weighted sum of this nodes 
					// contribution towards the error in the next layer
					double errorContrib = 0;
					
					// For each node in the next layer (forward layer)
					for(int j=0; j < nextBlames.length; j++) {
						// If this is a hidden layer
						if(current.next() != null) {
							double[] weights = current.next()
													.getWeightsAt(j);
							
							// add in blame from next layer node 
							// weighted by weight applied by that node 
							// to current layer node's output
							errorContrib += weights[i] * nextBlames[j];
						
						
						} else { // If this is an output layer, 
							// no weights to apply
							errorContrib += nextBlames[j];
						}
					}
					
					// Compute and modify delta
					//currentBlames[i] += outputs[i]*(1-outputs[i])*errorContrib;
					currentBlames[i] += act.dydk(i, outputs, i)*errorContrib;
				}
				
				// This layers inputs are previous-layer's outputs
				outputs = current.getLastInputs();
				nextBlames = currentBlames;
				
				// Moving on (backwards)
				current = current.prev();
				layerIdx--;
			}
		}
		
		/**
		 * Computes the new delta weights for each node in each layer
		 * Uses a running average.
		 */
		private void updateDeltaWeights() {
			// Update deltaWeights from head to tail
			Mlp.Layer current = mMlp.getHead();
			while(current != null) {
				
				double[] currentLayerBlames = current.getBlames();
				double[] inputs = current.getLastInputs();
			
				// For each node in this layer
				for(int i=0; i < current.size(); i++) {
					// Compute new weights according to:
					// weights' = weights + blame[i]*inputs
					// Note: We don't scale by learning rate here,
					// that is done when we compile it all in the main thread
					double blame = currentLayerBlames[i];
					double[] scaledInputs = Vector.scaled(inputs, blame);
					double[] deltaWeights = current.getDeltaWeightsAt(i);
					double[] cachedDeltaWeights = 
							new double[deltaWeights.length];
					System.arraycopy(deltaWeights, 0, cachedDeltaWeights, 0, deltaWeights.length);
					// AddTo performs addition in-place for performance
					Vector.addTo(scaledInputs, deltaWeights);
				}
				
				// Reset blames
				Arrays.fill(currentLayerBlames, 0);
				
				// Moving on
				current = current.next();
			}
		}
		
		/**
		 * Divides the delta weights by the numDataProcessed
		 */
		private void averageDeltaWeights() {
			
			if(numDataProcessed > 0) {
				
				// Divide delta-weights by numDataProcessed
				Mlp.Layer current = mMlp.getHead();
				while(current != null) {
				
					// For each node in this layer
					for(int i=0; i < current.size(); i++) {
						double[] deltaWeights = current.getDeltaWeightsAt(i);
						// AddTo performs addition in-place for performance
						double alpha = (1D/(double)numDataProcessed);
						Vector.scale(deltaWeights, alpha);
					}
					
					// Moving on
					current = current.next();
				}
			}
		}
		
	} // End TrainingTask
	
	
	
} // End NeuralNetworkTrainer
