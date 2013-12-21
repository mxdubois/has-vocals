import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

/**
 * A Multi-layer Perceptron. 
 * The Mlp is arranged as a linked list of Layers.
 * @author Michael DuBois
 *
 */
public class Mlp {

	//-------------------------------------------------------------------------
	// Static stuff
	//-------------------------------------------------------------------------
	
	private static Random random = new Random(System.currentTimeMillis());
	private static final double INITIAL_WEIGHT_MIN = 0.2;
	private static final double INITIAL_WEIGHT_MAX = 0.8;
	
	/**
	 * Loads an Mlp from a file. TODO
	 * @param file
	 * @param t
	 * @return
	 */
	public static Mlp loadFromFile(File file) 
		throws IOException, FileNotFoundException
	{
		// TODO Read values from file
		int numLayers = 0;
		double[][][] mlpArray = new double[numLayers][][];
		return new Mlp(mlpArray);
	}
	
	/**
	 * Writes an Mlp to a file. TODO
	 * @param mlp
	 * @param file
	 * @throws IOException
	 */
	public static void writeToFile(Mlp mlp, File file) 
		throws IOException
	{
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
	 * Generates a random double in the range (min, max) TODO refactor to utils
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
	
	//-------------------------------------------------------------------------
	// Instance stuff
	//-------------------------------------------------------------------------
	
	public Layer mHead;
	public Layer mTail;
	public int mSize;
	
	/**
	 * Creates a new Mlp.
	 * @param numNodes
	 */
	public Mlp() {
	}
	
	/**
	 * Copies an mlp
	 * @param mlp - the mlp to copy
	 */
	public Mlp(Mlp mlp) {
		Layer current = mlp.getHead();
		while(current != null) {
			append(new Layer(current));
			current = current.next();
		}
	}

	// TODO implement layers nodes weights
	public Mlp(double[][][] layersNodesWeights) {
		
	}
	
	/**
	 * Convenience method for pre-trained production mlps.
	 * @param inputs
	 * @return
	 */
	public double[] evaluate(double[] inputs) {
		return evaluate(inputs, false);
	}
	
	/**
	 * Returns the mlp's evaluation of this input
	 * @param inputs - the feature vector to evaluate
	 * @param isTraining
	 * @return
	 */
	public double[] evaluate(double[] inputs, boolean isTraining) {
		return mHead.evaluate(inputs, true, true, isTraining);
	}
	
	/**
	 * Inserts a layer at the end of the mlp.
	 * @param layer
	 * @return
	 */
	public Mlp append(Layer layer) {
		insertAt(mSize, layer);
		return this;
	}
	
	/**
	 * Inserts a layer at the beginning of the mlp.
	 * @param layer
	 * @return
	 */
	public Mlp prepend(Layer layer) {
		insertAt(0, layer);
		return this;
	}
	
	
	/**
	 * Inserts a layer into the mlp at given index
	 * @param idx
	 * @param layer
	 * @return
	 */
	public Mlp insertAt(int idx, Layer layer) {
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
		
			Layer current;
			if(idx > mSize/2) {
				// Iterate from tail
				current = mTail;
				int diff = mSize - idx;
				while(diff > 0) {
					current = current.prev();
					--diff;
				}
				
				// Insert and update all the linkages
				Layer next = current.next();
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
				
				// Insert and update all the linkages
				Layer prev = current.prev();
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
	
	/**
	 * Returns the ("hidden") layer at the front of this mlp.
	 * @return
	 */
	public Layer getHead() {
		return mHead;
	}
	
	/**
	 * Returns the output layer.
	 * @return
	 */
	public Layer getTail() {
		return mTail;
	}
	
	/**
	 * Returns the number of layers in this mlp.
	 * @return
	 */
	public int size() {
		return mSize;
	}
	
	@Override
	public String toString() {
		String str = "";
		Layer current = getHead();
		int i = 0;
		while(current !=null) {
			str += "== Layer " + i + " ===================\n";
			str += current.toString() + "\n";
			
			i++;
			current = current.next();
		}
		return str;
	}
	
	/**
	 * A single layer in the Mlp.
	 * @author Michael DuBois
	 *
	 */
	public static class Layer {
		
		private Layer mNext = null;
		private Layer mPrev = null;
		private Mlp.IActivationFunction mActivationFunction = null;
		
		// Nodes arranged as [nodeIdx][weights/deltaWeights][values]
		// TODO should probably just bite the bullet and use Node objects.
		private double[][][] mNodes = null;
		
		// Training vars
		private double[] mBlames = null;
		private double[] mLastInputs = null;
		
		/**
		 * Constructs a new layer.
		 * @param numNodes - the number of nodes the layer should have
		 * @param func - the activation function applied to all nodes in layer
		 */
		public Layer(int numNodes, IActivationFunction func) {
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
		 * Copies a Layer by values
		 * @param layer
		 */
		public Layer(Layer layer) {
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
		
		/**
		 * Returns this layer's evaluation of inputs with the option of 
		 * recursing through next layers. One might set recurse to false 
		 * to test or print the layer's output for a given input.
		 * @param inputs - the feature vector
		 * @param recurse - whether or not to recurse through next layers
		 * @param activation - activation function applied to layer's outputs
		 * @param isTraining - whether or not to track training variables
		 * @return
		 */
		public double[] evaluate(double[] inputs, 
								 boolean recurse, 
								 boolean activation, 
								 boolean isTraining) 
		{
			double [] outputs = inputs;
			
			// For performance, only cache last inputs when we're training
			if(isTraining) {
				mLastInputs = inputs;
				mBlames = new double[mNodes.length];
			}
			
			// If there are nodes, 
			if(mNodes.length > 0) {
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
					
					// Get a handle on the weights array
					double[] weights = mNodes[i][0];
					
					// weighted sum of weights and inputs
					outputs[i] = Vector.dot(weights,inputs);
					
					// If there is an activation function, run output through
					if(activation && mActivationFunction != null)
						outputs[i] = mActivationFunction.y(outputs, i);
				}
			} else if(activation && mActivationFunction != null){
				// if there are no nodes, but there is an activation func
				// we just return the activation of inputs
				for(int i=0; i < outputs.length; i++)
					outputs[i] = mActivationFunction.y(outputs, i);
			}
			
			// If caller wants the final output, recurse through next layers
			if(recurse && mNext != null)
				outputs = mNext.evaluate(outputs, recurse, activation, isTraining);
			
			return outputs;
		}
		
		/**
		 * Sets the blame vector for this layer.
		 * @param blames
		 */
		public void setBlames(double[] blames) {
			mBlames = blames;
		}
		
		/**
		 * Sets the weights vector at the given node
		 * @param idx - the index of the node
		 * @param weights - the new weights vector
		 */
		public void setWeightsAt(int idx, double[] weights) {
			mNodes[idx][0] = weights;
		}
		
		/**
		 * Sets the deltaWeights vector at the given node
		 * @param idx - the index of the node
		 * @param deltaWeights - the new deltaWeights
		 */
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
		
		/**
		 * Returns the blames vector for this layer
		 * @return
		 */
		public double[] getBlames() {
			return mBlames;
		}
		
		/**
		 * Returns the weights vector for a given node
		 * @param idx
		 * @return
		 */
		public double[] getWeightsAt(int idx) {
			return mNodes[idx][0];
		}
		
		/**
		 * Returns the deltaWeights for a given node
		 * @param idx
		 * @return
		 */
		public double[] getDeltaWeightsAt(int idx) {
			return mNodes[idx][1];
		}
		
		/**
		 * Returns the next layer
		 * @return next Layer or null if there is none
		 */
		public Layer next() {
			return mNext;
		}
		
		/**
		 * Returns the previous layer
		 * @return next Layer or null if there is none
		 */
		public Layer prev() {
			return mPrev;
		}
		
		/**
		 * Sets the previous layer
		 * @param layer
		 */
		private void setPrev(Layer layer){
			mPrev = layer;
		}
		
		/**
		 * Sets the next layer
		 * @param layer
		 */
		private void setNext(Layer layer){
			mNext = layer;
		}
		
		/**
		 * Returns the number of nodes in this layer.
		 * @return
		 */
		public int size() {
			return mNodes.length;
		}
		
		@Override
		public String toString() {
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

		/**
		 * Returns the IActivationFunction applied to this layer's outputs
		 * @return
		 */
		public IActivationFunction getActivationFunction() {
			return mActivationFunction;
		}
		
	} // End Layer
}
