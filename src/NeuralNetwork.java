import java.io.File;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

public class NeuralNetwork {

	public NeuralLayer mHead;
	public NeuralLayer mTail;
	public int mSize;
	
	/**
	 * Creates a new NeuralNetwork layer with numNodes nodes.
	 * @param numNodes
	 */
	public NeuralNetwork() {
	}
	
	/**
	 * Copies a neural network
	 * @param net - the neural network to copy
	 */
	public NeuralNetwork(NeuralNetwork net) {
		NeuralLayer current = net.getHead();
		while(current != null) {
			append(new NeuralLayer(current));
			current = current.next();
		}
	}

	// TODO implement layers nodes weights
	public NeuralNetwork(double[][][] layersNodesWeights, IActivationFunction t) {
		
	}
	
	/**
	 * Convenience method
	 * @param inputs
	 * @return
	 */
	public double[] evaluate(double[] inputs, boolean activation) {
		return evaluate(inputs, activation, false);
	}
	
	public double[] evaluate(double[] inputs, boolean activation, boolean isTraining) {
		return mHead.evaluate(inputs, activation, isTraining);
	}
	
	public NeuralNetwork append(NeuralLayer layer) {
		insertAt(mSize, layer);
		return this;
	}
	
	// TODO test/debug
	public NeuralNetwork prepend(NeuralLayer layer) {
		insertAt(0, layer);
		return this;
	}
	
	
	public NeuralNetwork insertAt(int idx, NeuralLayer layer) {
		if(idx > mSize || idx < 0) {
			throw new IndexOutOfBoundsException();
		}
		
		// TODO refactor/simplify if possible
		if(mHead == null) {
			mHead = layer;
			mTail = mHead;
		} else if(idx == 0) { 
			layer.setNext(mHead);
			mHead.setPrev(layer);
			mHead = layer;
		} else if (idx == mSize) { 
			layer.setPrev(mTail);
			mTail.setNext(layer);
			mTail = layer;
		}else {
		
			NeuralLayer current;
			if(idx > mSize/2) {
				// Iterate from tail
				current = mTail;
				int diff = mSize - idx;
				while(diff > 0) {
					current = current.prev();
					--diff;
				}
				
				// Update all the linkages
				NeuralLayer next = current.next();
				current.setNext(layer);
				layer.setNext(next);
				layer.setPrev(current);
				if(next != null)
					next.setPrev(layer);
			} else {
				// Iterate from head
				current = mHead;
				int diff = idx;
				while(diff > 0) {
					current = current.next();
					--diff;
				}
				
				// Update all the linkages
				NeuralLayer prev = current.prev();
				current.setPrev(layer);
				layer.setNext(current);
				layer.setPrev(prev);
				if(prev != null)
					prev.setNext(layer);
			}
		}
		
		mSize++;
		return this;
	}
	
	public NeuralLayer getHead() {
		return mHead;
	}
	
	public NeuralLayer getTail() {
		return mTail;
	}
	
	public int size() {
		return mSize;
	}
	
	@Override
	public String toString() {
		String str = "";
		NeuralLayer current = getHead();
		int i = 0;
		while(current !=null) {
			System.out.println("loopin around in the neural net");
			str += "== Layer " + i + " ===================\n";
			str += current.toString() + "\n";
			
			i++;
			current = current.next();
		}
		return str;
	}
	
	public static class NeuralLayer {
		
		private NeuralLayer mNext = null;
		private NeuralLayer mPrev = null;
		private NeuralNetwork.IActivationFunction mActivationFunction = null;
		private double[][][] mNodes = null;
		
		// Training vars
		private double[] mBlames = null;
		private double[] mLastInputs = null;
		
		public NeuralLayer(int numNodes, IActivationFunction func) {
			if(numNodes <=0) {
				throw new IllegalArgumentException(
						"numNodes must be greater than zero.");
			}
			mNodes = new double[numNodes][][];
			for(int i=0; i < mNodes.length; i++)
				mNodes[i] = new double[2][];
			mActivationFunction = func;
		}
		
		/**
		 * Copies a NeuralLayer
		 * @param layer
		 */
		public NeuralLayer(NeuralLayer layer) {
			mNodes = new double[layer.mNodes.length][][];
			// Copy the weights for each node
			for(int i=0; i<layer.mNodes.length; i++) {
				if(layer.mNodes[i] != null) {
					mNodes[i] = new double[layer.mNodes[i].length][];
					for(int j=0; j<layer.mNodes[i].length; j++) {
						if(layer.mNodes[i][j] != null) {
							mNodes[i][j]= new double[layer.mNodes[i][j].length];
							for(int k=0; k<layer.mNodes[i][j].length; k++)
								mNodes[i][j][k] = layer.mNodes[i][j][k];
						}
					}
				}
			}
			mActivationFunction = layer.mActivationFunction;
		}
		
		public double[] evaluate(double[] inputs, boolean activation, boolean isTraining) {
			double [] outputs = inputs;
	
			// For performance, only cache last inputs when we're training
			if(isTraining) {
				mLastInputs = inputs;
				mBlames = new double[mNodes.length];
			}
			
			// Generate an output vector from node outputs
			outputs = new double[mNodes.length];
			for(int i=0; i < mNodes.length; i++) {
				// If we haven't initialized weights for this node
				if(mNodes[i][0] == null) {
					// Do so now with this input vector's size
					mNodes[i][0] = newWeightsVector(inputs.length);
					int len = mNodes[i][0].length;
					mNodes[i][1] = new double[len];
				}
				double[] weights = mNodes[i][0];
				
				outputs[i] = Vector.dot(weights,inputs);
				
				if(activation && mActivationFunction != null)
					outputs[i] = mActivationFunction.y(outputs, i);
			}
			
			if(mNext != null)
				outputs = mNext.evaluate(outputs, activation, isTraining);
			return outputs;
		}
		
		public void setBlames(double[] blames) {
			mBlames = blames;
		}
		
		public void setWeightsAt(int idx, double[] weights) {
			mNodes[idx][0] = weights;
		}
		
		public void setDeltaWeightsAt(int idx, double[] deltaWeights) {
			mNodes[idx][1] = deltaWeights;
		}
		
		/**
		 * Returns the last inputs passed into this layer
		 * @return null if evaluate has not been called with isTraining == true
		 */
		public double[] getLastInputs() {
			return mLastInputs;
		}
		
		public double[] getBlames() {
			return mBlames;
		}
		
		public double[] getWeightsAt(int idx) {
			return mNodes[idx][0];
		}
		
		public double[] getDeltaWeightsAt(int idx) {
			return mNodes[idx][1];
		}
		
		public NeuralLayer next() {
			return mNext;
		}
		
		public NeuralLayer prev() {
			return mPrev;
		}
		
		private void setPrev(NeuralLayer layer){
			mPrev = layer;
		}
		
		private void setNext(NeuralLayer layer){
			mNext = layer;
		}
		
		public int size() {
			return mNodes.length;
		}
		
		@Override
		public String toString() {
			System.out.println("NeuralLayer toString");
			String str ="";
			str += "lastInputs: " + Arrays.toString(mLastInputs) + "\n";
			str += "Nodes: \n";
			for(int i=0; i < mNodes.length; i++) {
				str += "  Node " + i + "\n";
				if(mBlames != null)
					str += "    blame: " + mBlames[i] + "\n";
				str += "    weights: " + Arrays.toString(mNodes[i][0]) + "\n";
				str += "    deltaWeights: " + Arrays.toString(mNodes[i][1]) + "\n";
			}
			
			return str;
		}

		public IActivationFunction getActivationFunction() {
			return mActivationFunction;
		}
		
	} // End NeuralLayer

	
	//-------------------------------------------------------------------------
	// Static stuff
	//-------------------------------------------------------------------------
	private static Random random = new Random(System.currentTimeMillis());
	private static final double INITIAL_WEIGHT_MIN = .1;
	private static final double INITIAL_WEIGHT_MAX = .3;
	
	public static NeuralNetwork loadFromFile(File file, IActivationFunction t) {
		// TODO Read values from file
		int numLayers = 0;
		double[][][] networkArray = new double[numLayers][][];
		return new NeuralNetwork(networkArray, t);
	}
	
	public static void writeToFile(NeuralNetwork n, File file) {
	}
	
	/**
	 * Initializes a weights vector of given length with small random weights.
	 * @param length - length of the new weights vector
	 * @return
	 */
	public static double[] newWeightsVector(int length) {
		double[] weights = new double[length];
		for(int i=0; i < length; i++) {
			weights[i] = nextDouble(INITIAL_WEIGHT_MIN, 
								 INITIAL_WEIGHT_MAX, 
								 null);
		}
		return weights;
	}
	
	/**
	 * Generates a random double in the range (min, max)
	 * Gleaned from http://stackoverflow.com/a/3680648
	 * @param rangeMin
	 * @param rangeMax
	 */
	public static double nextDouble(double min, double max, Random r) {
		if(r == null)
			r = random;
		return min + (max - min) * r.nextDouble();
	}
	
	/**
	 * An interface for activation functions
	 * @author Michael DuBois
	 *
	 */
	public static interface IActivationFunction {
		public double[] y(double[] outputs);
		public double y(double[] output, int i);
		
		public double[] dydk(int k, double[] outputs);
		public double dydk(int k, double[] outputs, int i);
	}
}
