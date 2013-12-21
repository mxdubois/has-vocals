import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * A class to represent data with both features and labels.
 * @author Michael DuBois
 *
 */
public class LabeledData {

	double[] mFeatures;
	double[] mLabels;
	int mBaseFeatureLength;
	
	// Whether or not this data is fake
	boolean mIsFake = false;
	// The highest derivative that has been appended to the mFeatures array
	private int mHighestDerivative;
	
	/**
	 * Constructs a LabeledData object.
	 * @param features
	 * @param labels
	 */
	public LabeledData(double[] features, double[] labels) {
		this(features, features.length, 0, labels);
	}
	
	/**
	 * Constructs a LabeledData.
	 * @param features
	 * @param featureLength
	 * @param highestDerivative
	 * @param labels
	 */
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
	
	/**
	 * Set the features array and associated variables.
	 * @param features
	 * @param featureLength
	 * @param highestDerivative
	 */
	public void setFeatures(double[] features, int featureLength, int highestDerivative) {
		mFeatures = features;
		mBaseFeatureLength = featureLength;
		mHighestDerivative = highestDerivative;
	}
	
	/**
	 * Set the labels array.
	 * @param labels
	 */
	public void setLabels(double[] labels) {
		mLabels = labels;
	}
	
	/**
	 * Get the features array.
	 * @return
	 */
	public double[] getFeatures() {
		return mFeatures;
	}
	
	/*
	 * Get the labels array.
	 */
	public double[] getLabels(){
		return mLabels;
	}
	
	/**
	 * Get the length of the base features (without derivatives).
	 * @return
	 */
	public int baseFeatureLength() {
		return mBaseFeatureLength;
	}
	
	/**
	 * Get the number (1st, 2nd, etc) of the highest derivative appended to 
	 * features.
	 * @return
	 */
	public int highestDerivative() {
		return mHighestDerivative;
	}
	
	/**
	 * Set whether or not this data is fake data.
	 * @param isFake
	 */
	public void setIsFake(boolean isFake) {
		mIsFake = isFake;
	}
	
	/**
	 * Is this data fake?
	 * @return
	 */
	public boolean isFake() {
		return mIsFake;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(mFeatures) + Arrays.toString(mLabels);
	}
	
	/**
	 * Get the string representation of this object that should be written to
	 * a file.
	 * @return
	 */
	public String toFileString() {
		String str = "";
		str += mHighestDerivative + "\t";
		str += mBaseFeatureLength + "\t";
		str += ((mIsFake) ? 1 : 0) + "\t";
		
		for(double label : mLabels) {
			str += label + ",";
		}
		str += "\t";
		
		for(double feat : mFeatures) {
			str += feat + ",";
		}
		str += "\t";
		
		return str;
	}
	
	//--------------------------------------------------------------------------
	// STATIC STUFF
	//--------------------------------------------------------------------------
	
	public static final int FILE_POS_HIGHEST_DERIVATIVE = 0;
	public static final int FILE_POS_BASE_FEATURE_LENGTH = 1;
	public static final int FILE_POS_IS_FAKE = 2;
	public static final int FILE_POS_LABELS = 3;
	public static final int FILE_POS_FEATURES = 4;
	
	/**
	 * Write a list of Labeled Data to a file.
	 * @param list
	 * @param file
	 * @param append
	 * @return
	 * @throws IOException
	 */
	public static File writeToFile(List<LabeledData> list, File file, boolean append) 
			throws IOException 
	{
		BufferedWriter writer = null;
		
		try {
	        writer = new BufferedWriter(new FileWriter(file, append));
			for(LabeledData data : list) {
				writer.write(data.toFileString());
				writer.newLine();
			}
		    
		} finally {
		   try {writer.close();} catch (Exception ex) {}
		}
		return file;
	}
	
	/**
	 * Construct a LabeledData object from it's fileString representation.
	 * @param line
	 * @return
	 */
	public static LabeledData fromFileString(String line) {
		String[] items = line.split("\t");
		
		String[] labelStrings = items[FILE_POS_LABELS].split(",");
		double[] labels = new double[labelStrings.length];
		for(int i=0; i < labelStrings.length; i++) {
			labels[i] = Double.parseDouble(labelStrings[i]);
		}
		
		String[] featStrings = items[FILE_POS_FEATURES].split(",");
		double[] features = new double[featStrings.length];
		for(int i=0; i < featStrings.length; i++) {
			features[i] = Double.parseDouble(featStrings[i]);
		}
		
		int isFakeInt = Integer.parseInt(items[FILE_POS_IS_FAKE]);
		boolean isFake = (isFakeInt == 1);
		
		int highestDerivative = 
				Integer.parseInt(items[FILE_POS_HIGHEST_DERIVATIVE]);
		
		int baseFeatureLength = 
				Integer.parseInt(items[FILE_POS_BASE_FEATURE_LENGTH]);
		
		return new LabeledData(features, 
								baseFeatureLength, 
								highestDerivative, 
								labels);
	}
	
	/**
	 * Read an array of LabeledData from a file.
	 * @param file
	 * @return
	 */
	public static LabeledData[] readFromFile(File file) {
		BufferedReader reader = null;
		List<LabeledData> dataList = new ArrayList<LabeledData>();
		try {
			reader = new BufferedReader(new FileReader(file));

			String line = reader.readLine();
			while (line != null) {
				dataList.add(fromFileString(line));
				line = reader.readLine();
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		LabeledData[] array = new LabeledData[dataList.size()];
		return dataList.toArray(array);
	}
	
	/**
	 * A quick file read/write test
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		if(args.length <=0) {
			System.out.println("Please pass in an output filepath.");
			System.exit(1);
		}
		
		List<LabeledData> dataList = new ArrayList<LabeledData>();
		double[] labels1 = new double[3];
		labels1[0] = 0;
		labels1[1] = 1;
		labels1[2] = 2;
		double[] feats1 = new double[5];
		feats1[0] = 0;
		feats1[1] = 1;
		feats1[2] = 2;
		feats1[3] = 3;
		feats1[4] = 4;
		dataList.add(new LabeledData(feats1, 5, 0, labels1));
		
		double[] labels2 = new double[3];
		labels2[0] = .5;
		labels2[1] = 1.5;
		labels2[2] = 2.5;
		double[] feats2 = new double[4];
		feats2[0] = -0.5;
		feats2[1] = -1.5;
		feats2[2] = -2.5;
		feats2[3] = -3.5;
		LabeledData datum = new LabeledData(feats2, 4, 0, labels2);
		datum.setIsFake(true);
		dataList.add(datum);
		
		File file = new File(args[0]);
		writeToFile(dataList, file, false);
		LabeledData[] returnedArray = readFromFile(file);
		
		System.out.println("original list:");
		System.out.println(dataList.toString());
		System.out.println("returned list:");
		System.out.println(Arrays.toString(returnedArray));
		
	}
}
