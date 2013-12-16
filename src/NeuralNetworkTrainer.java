import java.io.PrintStream;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import com.sun.xml.internal.ws.util.StringUtils;


public class NeuralNetworkTrainer {


	
	// CLI Output formatting
	public static final int CONSOLE_WIDTH = 80;
	public static final char PROGRESS_CHAR = '=';
	public static final char PROGRESS_EDGE_CHAR = '|';
	public static final int NUM_CELL_SIZE = 5;
	
	public static void main(String[] args) {
		
	}
	
	private NeuralNetwork mMainNet;
	TrainDeltasTask[] mDeltaTasks;
	NeuralNetwork.NeuralLayer[] mThreadLayers;
	private PrintStream mOut;
	private int mEpoch = 0;
	private double mLastError = Double.NaN;
	private double mLearningRate = 1;
	
	public NeuralNetworkTrainer(NeuralNetwork net, PrintStream out) {
		NeuralNetwork mainNet = net;
		mOut = out;
	}
	
	public static String paddedString(char character, int length) {
		char[] array = new char[length];
	    Arrays.fill(array, PROGRESS_CHAR);
	    return new String(array);
	}
	
	public static String paddedCell(String value, int size) {
		return value + paddedString(' ', size - value.length());
	}
	
	public void updateStatus(double progress) {
		String status = "Last Error: ";
		status += paddedCell("" + mLastError, NUM_CELL_SIZE);
		status += "| Epoch: ";
		status += paddedCell("" + mEpoch, NUM_CELL_SIZE);
		
		int progressBarLength = CONSOLE_WIDTH - 2 - status.length();
		int segments = (int) (progress * (double) (progressBarLength));
		// Left pad with PROGRESS_CHAR
		String progressBar = paddedString(PROGRESS_CHAR, segments);
		progressBar += paddedString(' ', progressBarLength - segments);
	    status += PROGRESS_EDGE_CHAR + progressBar + PROGRESS_EDGE_CHAR;
		mOut.print(status + "\r");
	}
	
	public int trainNetwork(IDataContainer[] trainingContainers,
							IDataContainer[] testingContainers, 
								   double learningRate, 
								   double maxError,
								   int maxEpochs,
								   int numThreads) 
	{
		mEpoch = 0;
		mLearningRate = learningRate;
		
		ExecutorService executor = Executors.newFixedThreadPool(numThreads);
		
		// Create tasks
		mDeltaTasks = new TrainDeltasTask[numThreads];
		for(int i=0; i < mDeltaTasks.length; i++) {
			NeuralNetwork threadNet = new NeuralNetwork(mMainNet);
			// TODO create real subset
			IDataContainer[] subset = trainingContainers;
			mDeltaTasks[i] = new TrainDeltasTask(threadNet, subset);
		}
		
		mThreadLayers = new NeuralNetwork.NeuralLayer[mDeltaTasks.length];
		
		mOut.println("Beginning training session.");
		updateStatus(0);
		long startTime = System.currentTimeMillis();
		boolean converged = false;
		
		try {
			// While the network has not yet converged,
			while(!converged) {
				// Run threads
				for(int i=0; i < mDeltaTasks.length; i++) {
					executor.execute(mDeltaTasks[i]);
				}
				
				// Wait for them all to finish
				while(executor.awaitTermination(5, TimeUnit.MILLISECONDS)) {
					// Update the progress bar
					updateStatus(0);
					Thread.sleep(25);
				}
				
				// Accumulate weights to mMainNet and propagate back to threads
				adjustWeights();
				
				mLastError = test(testingContainers);
				converged = (mLastError <= maxError || mEpoch > maxEpochs);
				
				mEpoch++;
	
			}
		} catch(InterruptedException e) {
			e.printStackTrace();
		}
		
		long elapsed = System.currentTimeMillis() - startTime;
		mOut.println("Training complete. "
						+ mEpoch + " epochs. " 
						+ elapsed + "ms.");
		
		return mEpoch;
	}
	
	public void adjustWeights() {
		
		NeuralNetwork.NeuralLayer current;
		current = mMainNet.getHead();
		
		// Reset threadLayers to head... and set phasers to stun!
		for(int i=0; i < mThreadLayers.length; i++) {
			NeuralNetwork net = mDeltaTasks[i].getNetwork();
			mThreadLayers[i] = net.getHead();
		}
		
		// Adjust weights from head to tail and copy to all threads
		while(current != null) {
			// For each node in this layer
			for(int i=0; i < current.size(); i++) {
				
				double[] deltaWeights = current.getDeltaWeightsAt(i);
				
				// Accumulate deltaWeights from each thread
				for(int j=0; j < mThreadLayers.length; j++) {
					double[] threadDeltaWeights = 
							mThreadLayers[j].getDeltaWeightsAt(i);
					Vector.addTo(threadDeltaWeights, deltaWeights);
					
					// Reset deltaWeights for this node in thread
					Arrays.fill(deltaWeights, 0);
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
	
	private double test(IDataContainer[] testingContainers) {
		int trials = 0;
		double sumError = 0;
		for(IDataContainer dataContainer : testingContainers) {
			
				LabeledData datum;
				
				while(dataContainer.hasNext()) {
					try {
						datum = dataContainer.next();
						
						double[] outputs = mMainNet.evaluate(datum.getFeatures());
						double[] targets = datum.getLabels();
						sumError += computeError(outputs, targets);
						trials++;
					} catch(Exception e) {
						e.printStackTrace();
					}
				}
		}
		
		return (sumError / (double) trials);
	}
	
	public static double computeError(double[] outputs, double[] targets) {
		double[] diff = Vector.sub(outputs, targets);
		double summedSquares = Vector.dot(diff, diff);
		return .5 * summedSquares;
	}
	
	
	private class TrainDeltasTask implements Runnable {

		NeuralNetwork mNet;
		IDataContainer[] mDataContainers;
		
		TrainDeltasTask(NeuralNetwork net, IDataContainer[] subset) {
			mNet = net;
			mDataContainers = subset;
		}
		
		public NeuralNetwork getNetwork() {
			return mNet;
		}
		
		@Override
		public void run() {
			for(IDataContainer dataContainer : mDataContainers){
				LabeledData datum;
				while(dataContainer.hasNext()) {
					
					try {
						datum = dataContainer.next();
					
						double[] outputs = mNet.evaluate(datum.getFeatures(), true);
						double[] targets = datum.getLabels();
						
	
						computeBlames(outputs, targets);
						updateDeltaWeights();
					
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					
				}
			}
		}
		
		/**
		 * Computes and assigns a blame factor vector for each layer
		 * In academic papers this is often denoted with a lowercase delta
		 * @param outputs
		 * @param targets
		 */
		private void computeBlames(double[] outputs, double[] targets) {
			// For output nodes, the blame is just assumed to be the 
			// difference between outputs and targets
			double[] nextBlames = Vector.sub(targets, outputs);
			double[] currentBlames;
			
			// Back-propagate from tail
			NeuralNetwork.NeuralLayer current = mNet.getTail();
			while(current != null) {
				currentBlames = current.getBlames();
				
				// For each node in this layer
				for(int i=0; i < outputs.length; i++) {
					
					double output = outputs[i];
					// The errorContrib is a weighted sum of this nodes 
					// contribution towards the error in the next layer
					double errorContrib = 0;
					
					// For each node in the next layer (forward layer)
					for(int j=0; j < nextBlames.length; j++) {
						// If this is a hidden layer
						if(current.next() != null) {
							double[] weights = current.next()
													.getWeightsAt(j);
							
							// add in delta from next layer node 
							// weighted by weight applied by that node 
							// to current layer node's output
							errorContrib += weights[i] * nextBlames[i];
						
						
						} else { // If this is an output layer, 
							// no weights to apply
							errorContrib += nextBlames[j];
						}
					}
					
					// Compute and modify delta
					currentBlames[i] += output*(1-output)*errorContrib;
					
				}
				
				// This layers inputs are previous-layer's outputs
				outputs = current.getLastInputs();
				nextBlames = currentBlames;
				
				// Moving on (backwards)
				current = current.prev();
			}
		}
		
		/**
		 * Computes the new delta weights for each node in each layer
		 */
		private void updateDeltaWeights() {
			// Update deltaWeights from head to tail
			NeuralNetwork.NeuralLayer current = mNet.getHead();
			while(current != null) {
				
				double[] currentLayerDeltas = current.getBlames();
				double[] inputs = current.getLastInputs();
			
				// For each node in this layer
				for(int i=0; i < current.size(); i++) {
					// Compute new weights according to:
					// weights' = weights + learningRate*delta[i]*inputs
					double alpha = mLearningRate * currentLayerDeltas[i];
					double[] scaledInputs = Vector.scaled(inputs, alpha);
					double[] deltaWeights = current.getDeltaWeightsAt(i);
					// AddTo performs addition in-place for performance
					Vector.addTo(scaledInputs, deltaWeights);
				}
				
				// Reset deltas
				Arrays.fill(currentLayerDeltas, 0);
				
				// Moving on
				current = current.next();
			}
		}
		
	} // End TrainDeltasTask
	
	
	
} // End NeuralNetworkTrainer
