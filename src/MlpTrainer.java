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
	TrainDeltasTask[] mDeltaTasks;
	Mlp.Layer[] mThreadLayers;
	private PrintStream mOut;
	private int mEpoch = 0;
	private double mLastError = Double.NaN;
	private double mLearningRate = 1;
	private List<Future<Mlp>> mFutures;
	private String mLastErrorStr;
	private double mLastErrorPercent;
	
	public MlpTrainer(Mlp net, PrintStream out) {
		mMainNet = net;
		mOut = out;
	}
	
	public static String paddedString(char character, int length) {
		char[] array = new char[length];
	    Arrays.fill(array, character);
	    return new String(array);
	}
	
	public static String paddedCell(String value, int size) {
		return value + paddedString(' ', size - value.length());
	}
	
	public void updateTrainingStatus(double progress) {
		progress = Math.max(Math.min(progress,1), 0);
		String status = "E: % ";
		status += paddedCell("" + mLastErrorStr, ERROR_CELL_SIZE);
		status += " | Epoch: ";
		status += paddedCell("" + mEpoch, EPOCH_CELL_SIZE);
		printWithProgress(status, progress);
	}
	
	public void updateTestingStatus(double progress) {
		progress = Math.max(Math.min(progress,1), 0);
		String status = "Testing...";
		printWithProgress(status, progress);
	}
	
	public void printWithProgress(String status, double progress) {
		int progressBarLength = CONSOLE_WIDTH - 2 - status.length();
		int segments = (int) (progress * progressBarLength);
		// Left pad with PROGRESS_CHAR
		String progressBar = paddedString(PROGRESS_CHAR, segments);
		progressBar += paddedString(' ', (progressBarLength - segments));
	    status += PROGRESS_EDGE_CHAR + progressBar + PROGRESS_EDGE_CHAR;
		mOut.print(status + "\r");
	}
	
	public void prepareNetwork(IDataContainer[] trainingContainers) {
		try {
			System.out.println("opening first container");
			trainingContainers[0].open();
			System.out.println("container opened");
			LabeledData datum = trainingContainers[0].next();
			if(datum != null){
				double[] outputs = 
					mMainNet.evaluate(datum.getFeatures(), true);
				mOut.println("Outputs: " + Arrays.toString(outputs));
			} else {
				throw new IllegalStateException(
						"First container wouldn't open. duhdur.");
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			try { trainingContainers[0].close();} catch (Exception e) {}
		}
	}
	
	public int trainNetwork(IDataContainer[] trainingContainers,
							IDataContainer[] testingContainers, 
								   double maxError,
								   int maxEpochs,
								   int maxThreads) 
	{
		mEpoch = 0;
		mLastError = 100;
		
		// Determine the correct number of threads
		int numThreads = Runtime.getRuntime().availableProcessors();
		numThreads = Math.min(numThreads, maxThreads);
		numThreads = Math.min(numThreads, trainingContainers.length);
		
		// Determine how to split the data among threads
		int containersPerThread = trainingContainers.length / numThreads;
		System.out.println("Using " + numThreads + " threads!");
		mOut.println("containersPerThread: " + containersPerThread);
		
		// Make sure the network is prepared for given data
		prepareNetwork(trainingContainers);
		
		// We use a completion service to manage callable threads
		ExecutorService executor = Executors.newFixedThreadPool(numThreads);
		CompletionService<Mlp> ecs = 
				new ExecutorCompletionService<Mlp>(executor);
		
		
		// Arrays to keep track of threads
		mDeltaTasks = new TrainDeltasTask[numThreads];
		mFutures = new ArrayList<Future<Mlp>>();
		mThreadLayers = new Mlp.Layer[mDeltaTasks.length];
		
		// Create the callables
		for(int i=0; i < mDeltaTasks.length; i++) {
			// thread's offset in data container list
			int cOffset = containersPerThread * i;
			int cEndIdx = cOffset + containersPerThread;
			
			// Get subset of data for thread
			IDataContainer[] subset =  new IDataContainer[containersPerThread];
			List<IDataContainer> subsetList = Arrays.asList(trainingContainers);
			subsetList.subList(cOffset, cEndIdx).toArray(subset);
			
			// Create the task with it's own copy of the network
			Mlp threadNet = new Mlp(mMainNet);
			mDeltaTasks[i] = new TrainDeltasTask(threadNet, subset);
		}
		
		mOut.println("Beginning training session...");
		long startTime = System.currentTimeMillis();

		try {
			
			// compute the initial error for reference
			computeError(testingContainers);
			
			// While the network has not yet converged,
			boolean converged = false;
			while(!converged) {
				// Update the learning rate with decay
				// TODO expose decay as a config var
				mLearningRate = 1D / (.01*mEpoch + 1D);
				
				updateTrainingStatus(0); // 0% progress
				long epochStart = System.currentTimeMillis();
				
				mFutures.clear();
				
				// Submit threads to executor
				for(int i=0; i < mDeltaTasks.length; i++) {
					ecs.submit(mDeltaTasks[i]);
				}
				
				// Wait for them all to finish
				long sleepTime = SLEEP_TIME_START;
				while(mFutures.size() < numThreads) {
					Future<Mlp> future = 
							ecs.poll(1, TimeUnit.NANOSECONDS);
					
					if(future != null) {
						sleepTime = SLEEP_TIME_START;
						mFutures.add(future);
					} else {
						sleepTime += SLEEP_TIME_INCREMENT;
						sleepTime = Math.min(sleepTime, SLEEP_TIME_MAX);
					}
					
					// Update the progress bar
					double processed = 0;
					for(TrainDeltasTask task : mDeltaTasks) {
						processed += task.getNumProcessed();
					}
					updateTrainingStatus(processed / trainingContainers.length);
					Thread.sleep(sleepTime);
				}
				mOut.println();
				
				long elapsed = System.currentTimeMillis() - epochStart;
				mOut.println("Epoch complete. " + elapsed + "ms.");
				mOut.println(" Compiling results...");
				long compilingStart = System.currentTimeMillis();
				
				// Accumulate weights to mMainNet and propagate back to threads
				adjustWeights();
				
				elapsed = System.currentTimeMillis() - compilingStart;
				mOut.println("Finished compiling results." 
								+ elapsed + "ms.");
				
				
				computeError(testingContainers);
				
				
				if(mLastError <= maxError) {
					converged = true;
				} else if(mEpoch >= maxEpochs) {
					converged = true;
					mOut.println("WARNING: Exceeded max epochs.");
				}
				mEpoch++;
	
			}
		} catch(InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		executor.shutdownNow();
		long elapsed = System.currentTimeMillis() - startTime;
		mOut.println("Training complete. "
						+ mEpoch + " epochs. " 
						+ elapsed + "ms.");
		mOut.println("--- Your Neural Network ---");
		mOut.println(mMainNet.toString());
		
		return mEpoch;
	}
	
	public void adjustWeights() throws ExecutionException, InterruptedException {
		
		Mlp.Layer current;
		current = mMainNet.getHead();
		
		// Reset threadLayers to head... and set phasers to stun!
		for(int i=0; i < mFutures.size(); i++) {
			Mlp net = mFutures.get(i).get();
			mThreadLayers[i] = net.getHead();
		}
		
		// Adjust weights from head to tail and copy to all threads
		while(current != null) {
			
			// For each node in this layer
			for(int i=0; i < current.size(); i++) {
				
				double[] deltaWeights = current.getDeltaWeightsAt(i);
				
				// Accumulate deltaWeights from each thread
				for(int j=0; j < mThreadLayers.length; j++) {
					//System.out.println("####### Thread " + j + " ######");
					//System.out.println(mThreadLayers[j].toString());
					double[] threadDeltaWeights = 
							mThreadLayers[j].getDeltaWeightsAt(i);
					Vector.scale(threadDeltaWeights, 1D /mFutures.size() );
					Vector.addTo(threadDeltaWeights, deltaWeights);
					
					// Reset deltaWeights for this node in thread
					Arrays.fill(threadDeltaWeights, 0);
				}
				
				
				// Scale deltaWeights in-place
				Vector.scale(deltaWeights, mLearningRate);
				double[] weights = current.getWeightsAt(i);
				// Adjust weights in-place
				Vector.addTo(deltaWeights, weights);
				
				// Copy weights for this node to each thread
				for(int j=0; j < mThreadLayers.length; j++) {
					
					double[] threadWeights = mThreadLayers[j].getWeightsAt(i);
					System.arraycopy(weights, 0, 
									 threadWeights, 0, 
									 weights.length);
				}
				
				// Reset deltaWeights for this node in main
				Arrays.fill(deltaWeights, 0);
			}
			
			// Advance all layers
			current = current.next();
			for(int i=0; i < mThreadLayers.length; i++) {
				mThreadLayers[i] = mThreadLayers[i].next();
			}
		}
	}
	
	private void computeError(IDataContainer[] testingContainers) {
		mOut.println("Computing error");
		long testingStart = System.currentTimeMillis();
		mLastError = test(testingContainers);
		mLastErrorPercent = mLastError * 100D;
		mLastErrorStr = "" + (Math.round(mLastErrorPercent * 10000D) / 10000D);
		long elapsed = System.currentTimeMillis() - testingStart;
		mOut.println("Error Computed (%" + mLastError + ")." 
				+ elapsed + "ms.");
	}
	
	/**
	 * Returns the main net's root mean square error on testing dataset.
	 * @param testingContainers
	 * @return
	 */
	private double test(IDataContainer[] testingContainers) {
		double sumSqrResiduals = 0;
		int trials = 0;
		int targetsLength = 0;
		
		for(IDataContainer dataContainer : testingContainers) {
			
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
		}
		double quantity = trials * targetsLength;
		return sumSqrResiduals / quantity;
	}
	
	private class TrainDeltasTask implements Callable<Mlp> {

		Mlp mNet;
		IDataContainer[] mDataContainers;
		AtomicInteger numContainersProcessed;
		int numDataProcessed = 0;
		
		TrainDeltasTask(Mlp net, IDataContainer[] subset) {
			mNet = net;
			mDataContainers = subset;
			numContainersProcessed = new AtomicInteger(0);
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
									mNet.evaluate(datum.getFeatures(), true);
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
			
			return mNet;
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
			
			int layerIdx = mNet.size() - 1;
			Mlp.Layer current = mNet.getTail();

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
			Mlp.Layer current = mNet.getHead();
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
				Mlp.Layer current = mNet.getHead();
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
		
		/**
		 * (thread-safe) Returns the number of data containers processed 
		 * @return
		 */
		public int getNumProcessed() {
			return numContainersProcessed.get();
		}

		
		
	} // End TrainDeltasTask
	
	
	
} // End NeuralNetworkTrainer
