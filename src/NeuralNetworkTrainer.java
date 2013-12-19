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


public class NeuralNetworkTrainer {


	
	// CLI Output formatting
	public static final int CONSOLE_WIDTH = 80;
	public static final char PROGRESS_CHAR = '=';
	public static final char PROGRESS_EDGE_CHAR = '|';
	public static final int NUM_CELL_SIZE = 20;
	
	// Sleep time in ms between checks if threads are complete
	private static final long SLEEP_TIME_START = 2;
	private static final long SLEEP_TIME_INCREMENT = 10;
	private static final long SLEEP_TIME_MAX = 150;
	
	public static void main(String[] args) {
		
	}
	
	private NeuralNetwork mMainNet;
	TrainDeltasTask[] mDeltaTasks;
	NeuralNetwork.NeuralLayer[] mThreadLayers;
	private PrintStream mOut;
	private int mEpoch = 0;
	private double mLastError = Double.NaN;
	private double mLearningRate = 1;
	private List<Future<NeuralNetwork>> mFutures;
	private String mLastErrorStr;
	
	public NeuralNetworkTrainer(NeuralNetwork net, PrintStream out) {
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
	
	public void updateStatus(double progress) {
		progress = Math.max(Math.min(progress,1), 0);
		String status = "E: %";
		status += paddedCell("" + mLastErrorStr, NUM_CELL_SIZE);
		status += " | Epoch: ";
		status += paddedCell("" + mEpoch, NUM_CELL_SIZE);
		
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
								   int maxEpochs) 
	{
		System.out.println("trainer called!");
		mEpoch = 0;
		mLastError = 100;
		int numThreads = Runtime.getRuntime().availableProcessors();
		numThreads = Math.min(numThreads, trainingContainers.length);
		
		prepareNetwork(trainingContainers);
		System.out.println("network prepared!");	
		ExecutorService executor = Executors.newFixedThreadPool(numThreads);
		System.out.println("executor initialized!");
		
		CompletionService<NeuralNetwork> ecs = 
				new ExecutorCompletionService<NeuralNetwork>(executor);
		
		// Create tasks
		mDeltaTasks = new TrainDeltasTask[numThreads];
		mFutures = new ArrayList<Future<NeuralNetwork>>();
		int containersPerThread = trainingContainers.length / numThreads;
		mOut.println("containersPerThread: " + containersPerThread);
		for(int i=0; i < mDeltaTasks.length; i++) {
			
			int cOffset = containersPerThread * i;
			int cEndIdx = cOffset + containersPerThread;
			
			// Get subset of data
			IDataContainer[] subset =  new IDataContainer[containersPerThread];
			List<IDataContainer> subsetList = Arrays.asList(trainingContainers);
			subsetList.subList(cOffset, cEndIdx).toArray(subset);
			
			NeuralNetwork threadNet = new NeuralNetwork(mMainNet);
			mDeltaTasks[i] = new TrainDeltasTask(threadNet, subset);
		}
		System.out.println("deltaTasks initialized!");
		
		mThreadLayers = new NeuralNetwork.NeuralLayer[mDeltaTasks.length];
		
		mOut.println("Beginning training session.");
		
		long startTime = System.currentTimeMillis();
		boolean converged = false;
		
		try {
			// While the network has not yet converged,
			while(!converged) {
				mFutures.clear();
				// TODO expose this as a config var
				mLearningRate = 1D / (1.5*mEpoch + 1D);
				// Run threads
				System.out.println("running " + numThreads + " threads!");
				updateStatus(0);
				long epochStart = System.currentTimeMillis();
				for(int i=0; i < mDeltaTasks.length; i++) {
					ecs.submit(mDeltaTasks[i]);
				}
				
				
				// Wait for them all to finish
				long sleepTime = SLEEP_TIME_START;
				while(mFutures.size() < numThreads) {
					Future<NeuralNetwork> future = 
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
					updateStatus(processed / trainingContainers.length);
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
				
				mOut.println("Computing error");
				long testingStart = System.currentTimeMillis();
				mLastError = test(testingContainers);
				mLastErrorStr = "" + (Math.round(mLastError * 10000D) / 10000D);
				elapsed = System.currentTimeMillis() - testingStart;
				mOut.println("Finished testing error (%" + mLastError + ")." 
						+ elapsed + "ms.");
				
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
		
		NeuralNetwork.NeuralLayer current;
		current = mMainNet.getHead();
		
		// Reset threadLayers to head... and set phasers to stun!
		for(int i=0; i < mFutures.size(); i++) {
			NeuralNetwork net = mFutures.get(i).get();
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
				//Arrays.fill(deltaWeights, 0);
			}
			
			// Advance all layers
			current = current.next();
			for(int i=0; i < mThreadLayers.length; i++) {
				mThreadLayers[i] = mThreadLayers[i].next();
			}
		}
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
		SimpleRegression regression = new SimpleRegression();
		
		for(IDataContainer dataContainer : testingContainers) {
			
				LabeledData datum;
				try {
					dataContainer.open();
					
					while(dataContainer.hasNext()) {
						
							datum = dataContainer.next();
							
							double[] targets = datum.getLabels();
							double[] feats = datum.getFeatures();
							double[] outputs = 
									mMainNet.evaluate(feats, true, true);
							for(int i=0; i < targets.length; i++) {
								regression.addData(targets[i], outputs[i]);
							}
							
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
		return regression.getMeanSquareError();
	}
	
	private class TrainDeltasTask implements Callable<NeuralNetwork> {

		NeuralNetwork mNet;
		IDataContainer[] mDataContainers;
		AtomicInteger numContainersProcessed;
		int numDataProcessed = 0;
		
		TrainDeltasTask(NeuralNetwork net, IDataContainer[] subset) {
			mNet = net;
			mDataContainers = subset;
			numContainersProcessed = new AtomicInteger(0);
		}
		
		public NeuralNetwork getNetwork() {
			return mNet;
		}
		
		@Override
		public NeuralNetwork call() throws Exception {
			numContainersProcessed.set(0);
			numDataProcessed = 0;
			for(IDataContainer dataContainer : mDataContainers){
				LabeledData datum;
				try {
					dataContainer.open();
					while(dataContainer.hasNext()) {
						
							datum = dataContainer.next();
						
							double[] outputs = 
									mNet.evaluate(datum.getFeatures(), true, true);
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
			//finishDeltaWeights();
			
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
			NeuralNetwork.NeuralLayer current = mNet.getTail();

			// Get the output layer's activation function
			NeuralNetwork.IActivationFunction act = 
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
					
					if(Double.isInfinite(currentBlames[i])) {
						throw new IllegalStateException("INFINITE BLAME MOFO"
									+ "\n LAYER: " + layerIdx
									+ "\n errorContrib: " + errorContrib
									+ "\n dydk: " + act.dydk(i, outputs, i)
									+ "\n outputs["+i+"]: " + outputs[i]
									+ "\n lastInputs: " + Arrays.toString(current.getLastInputs())
									+ "\n rawOutputs: " + Arrays.toString(current.evaluate(current.getLastInputs(), false, false))
								);
					}
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
			NeuralNetwork.NeuralLayer current = mNet.getHead();
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
					for(int j=0; j < deltaWeights.length; j++) {
						if(Double.isNaN(deltaWeights[j]) ) {
							throw new IllegalStateException("GOD DAMMIT! "
									+ "\n inputs: " + Arrays.toString(inputs)
									+ "\n before deltaWeights: " + Arrays.toString(cachedDeltaWeights)
									+ "\n scale (blame): " + blame
									+ "\n scaledInputs: " + Arrays.toString(scaledInputs)
									+ "\n after deltaWeights: " + Arrays.toString(deltaWeights)
									);
						}
					}
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
		private void finishDeltaWeights() {
			
			if(numDataProcessed > 0) {
				
				// Divide delta-weights by numDataProcessed
				NeuralNetwork.NeuralLayer current = mNet.getHead();
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
		
		public int getNumProcessed() {
			return numContainersProcessed.get();
		}

		
		
	} // End TrainDeltasTask
	
	
	
} // End NeuralNetworkTrainer
