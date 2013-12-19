
public class StandardLogistic implements Mlp.IActivationFunction {

	double mAlpha;
	
	public StandardLogistic(double alpha) {
		mAlpha = alpha;
	}
	
	@Override
	public double[] y(double[] outputs) {
		double[] vals = new double[outputs.length];
		// Overflow-safe equivalent of standard logistic function
		for(int i=0; i < outputs.length; i++)
			vals[i] = .5D * (1D + Math.tanh(mAlpha*.5D*outputs[0]));
		return vals;
	}

	@Override
	public double y(double[] outputs, int i) {
		return y(outputs)[i];
	}

	@Override
	public double[] dydk(int k, double[] outputs) {
		double[] vals = new double[outputs.length];
		for(int i=0; i < outputs.length; i++)
			vals[i] = dydk(k, outputs, i);
		return vals;
	}

	@Override
	public double dydk(int k, double[] outputs, int i) {
		return (1-outputs[i])*outputs[k];
	}

}
