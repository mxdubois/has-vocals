import java.util.Arrays;


public class LabeledData {

	double[] mFeatures;
	double[] mLabels;
	int mBaseFeatureLength;
	boolean mIsFake = false;
	private int mHighestDerivative;
	
	public LabeledData(double[] features, double[] labels) {
		this(features, features.length, 0, labels);
	}
	
	public LabeledData(double[] features, 
					   int featureLength, 
					   int highestDerivative, 
					   double[] labels) 
	{
		mLabels = labels;
		setFeatures(features, featureLength, highestDerivative);
	}
	
	/**
	 * Constructs a copy of given LabeledData
	 * @param data
	 */
	public LabeledData(LabeledData data) {
		double[] features = data.getFeatures();
		double[] labels = data.getLabels();
		mFeatures = new double[features.length];
		mLabels = new double[labels.length];
		System.arraycopy(features, 0, mFeatures, 0, features.length);
		System.arraycopy(labels, 0, mLabels, 0, labels.length);
		mIsFake = data.isFake();
		mBaseFeatureLength = data.baseFeatureLength();
		mHighestDerivative = data.highestDerivative();
	}
	
	public void setFeatures(double[] features, int featureLength, int highestDerivative) {
		mFeatures = features;
		mBaseFeatureLength = featureLength;
		mHighestDerivative = highestDerivative;
	}
	
	public void setLabels(double[] labels) {
		mLabels = labels;
	}
	
	public double[] getFeatures() {
		return mFeatures;
	}
	
	public double[] getLabels(){
		return mLabels;
	}
	
	public int baseFeatureLength() {
		return mBaseFeatureLength;
	}
	
	public int highestDerivative() {
		return mHighestDerivative;
	}
	
	public void setIsFake(boolean isFake) {
		mIsFake = isFake;
	}
	
	public boolean isFake() {
		return mIsFake;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(mFeatures) + Arrays.toString(mLabels);
	}
	
	
	public static void appendDerivatives(LabeledData[] data) 
	{
		
		int cidx = data.length / 2;
		int t = (data.length - 1) / 2;
		double[] currFeats = data[cidx].getFeatures();
		int length = currFeats.length;
		
		double[] newCurrFeats = new double[length*3];
		System.arraycopy(currFeats, 0, newCurrFeats, 0, length);
		
		for(int i=0; i < length; i++) {
			int j = i + length;
			double numerator =  0;
			double denominator = 0;
			for(int k=1; k < t; t++) {
				double[] prevFeats = data[cidx - k].getFeatures();
				double[] nextFeats = data[cidx + k].getFeatures();
				numerator +=  k * (nextFeats[i] - prevFeats[i]);
				denominator += ( k * k );
			}
			double delta = numerator / (2D * denominator);
			
			newCurrFeats[j] = 0;
		}
		
	}
	
}
