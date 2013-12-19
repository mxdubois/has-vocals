import java.util.Arrays;

/**
 * The SoftMax activation function. 
 * This devolves to the logistic function if outputs.length == 1.
 * @author Michael DuBois
 *
 */
public class SoftMax implements NeuralNetwork.IActivationFunction {
	
	private double mAlpha;
	private StandardLogistic mLogistic;
	
	public SoftMax(double alpha) {
		mAlpha = alpha;
		
		mLogistic = new StandardLogistic(mAlpha);
	}
	
	/**
	 * A overflow-safe soft-max function adapted from
	 * http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
	 */
	public double[] y(double[] outputs) {
		
		// If the output vector is length 1, devolve to logistic function
		if(outputs.length == 1) {
			return mLogistic.y(outputs);
		}
		
		double a = Vector.max(outputs);
		
		double Z = 0.0;
		for (int i = 0; i < outputs.length; ++i) {
			//System.out.println("i: " + i );
			double exponent =  mAlpha * (outputs[i] - a);
			//System.out.println("exponent: " + exponent);
			double result = Math.exp( exponent );
			//System.out.println("Adding " + result +  " to Z");
		    Z += result;
		}
		
		System.out.println("Z:" + Z);
		
		double[] ps = new double[outputs.length];
		for (int i = 0; i < outputs.length; ++i) 
				ps[i] = Math.exp( mAlpha * (outputs[i] - a) )/Z;
		
		return ps;
	}
	
	@Override
	public double y(double[] outputs, int i) {
		return y(outputs)[i];
	}
	
	@Override
	public double[] dydk(int k, double[] outputs) {
		
		// If the output vector is length 1, devolve to logistic function
		if(outputs.length == 1) {
			return mLogistic.dydk(k, outputs);
		}
		
		double[] deltas = new double[outputs.length];
		for(int i=0; i < outputs.length; i++) {
			double kronDelta = (i == k) ? 1 : 0;
			deltas[i] = (kronDelta - outputs[i]) * outputs[k];
		}
		
		return deltas;
	}
	
	@Override
	public double dydk(int k, double[] outputs, int i) {	
		return dydk(k, outputs)[i];
	}
	
	/**
	 * A quick test.
	 */
	public static void main(String[] args) {
		double[] alphas = new double[] { 1, .5, 1.7};
		
		SoftMax softMax;
		for(double alpha : alphas) {
			System.out.println("Testing SoftMax(" + alpha + ")");
			softMax = new SoftMax(alpha);
			
			// TEST 1
			double[] test1 = new double[] { 
					1,2,3
			};
			
			System.out.println("test1: "  
					+ Arrays.toString(test1));
			
			double[] softMaxed = softMax.y(test1);
			System.out.println("y(test1): "  
					+ Arrays.toString(softMaxed));
			
			System.out.println("dydk(0, dy(test1)): "  
					+ Arrays.toString(softMax.dydk(0, softMaxed)));
			
			// TEST 2
			double[] test2 = new double[] { 0};
			
			System.out.println("test2: "  
					+ Arrays.toString(test2));
			
			softMaxed = softMax.y(test2);
			System.out.println("y(test2): "  
					+ Arrays.toString(softMaxed));
			
			System.out.println("dydk(0, dy(test2)): "  
					+ Arrays.toString(softMax.dydk(0, softMaxed)));
		}
		
		
			
	}
}
