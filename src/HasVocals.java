import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import au.com.bytecode.opencsv.CSVReader;

public class HasVocals {

	public static final String TAG = "HasVocals";
	public static final int MAIN_REQUIRED_ARGS = 1;
	
	public static void main(String[] args) throws IllegalArgumentException {
		
		// Option defaults
		boolean recurse = false;
		int n = Integer.MAX_VALUE;
		int maxEpochs = 1000;
		int maxThreads = Integer.MAX_VALUE;
		double minDeltaError = .00001;
		
		File dataFile = null;
        File audioDir = null;
		
		// Things that could one day be options
		String[] filetypes = new String[] {"wav"};
		
		String helpStr = "Usage: " + TAG 
                + "[options] dataFile audioDir tempDir \n"
                + "-r|-R search audiodir recursively \n"
                + "-n|-N int, specify max # of songs to train on \n"
                + "-m|-M int, specify max # of epochs (default " 
                		+ maxEpochs + ") \n"
                + "-h display this help message";
		
		if(args.length < MAIN_REQUIRED_ARGS) {
           System.out.println("You didn't supply the required parameters." 
            			+ " Use option -h for help.");
           System.exit(1);
        }
		
		//Required args
        File temp = new File(args[args.length - 1]);
		
		if(args.length > MAIN_REQUIRED_ARGS) {
            // Get options
            for(int i=0; i < args.length - MAIN_REQUIRED_ARGS; i++) {
            	try {    
            		if(args[i].startsWith("-") && args[i].length() == 2) {
	                    // It is a valid option
	                    char option = args[i].charAt(1);
	                    switch(option) {
	                    case 'h':
	                        // Display help
	                        System.out.println(helpStr);
	                        System.exit(0);
	                        
	                    case 'r' :
	                    case 'R' :
	                        // Recurse on directories
	                        recurse = true;
	                        break;
	                        
	                    case 'm' :
	                    case 'M' :
	                    	maxEpochs = 
	                    		Integer.parseInt(getOptionParameter(args, i));
	                    	i++;
	                        break;
	                        
	                    case 'n' :
	                    case 'N' :
	                    	n = 
	                    		Integer.parseInt(getOptionParameter(args, i));
	                    	i++;
	                        break;
	                        
	                    case 'e' :
	                    case 'E' :
	                    	minDeltaError = Double.parseDouble( 
	                    					getOptionParameter(args, i) );
	                    	i++;
	                        break;
	                        
	                    case 'd' :
	                    case 'D' :
	                    	String dataPath = getOptionParameter(args, i);
	                    	String audioPath = getOptionParameter(args, i + 1);
	                    	dataFile = new File(dataPath);
	                        audioDir = new File(audioPath);
	                    	i+=2;
	                        break;
	                        
	                    case 't' :
	                    case 'T' :
	                    	maxThreads = 
	                    		Integer.parseInt(getOptionParameter(args, i));
	                    	i++;
	                        break;
	                        
	                    default :
	                    	System.out.println("Invalid flag " + args[i]
	                                			+ "Use option -h for help.");
	                    	System.exit(1);
	                    }
	                } else {
	                   System.out.println(args[i] + 
	                		   " looks like a malformed option.\n"
	                            + "Use option -h for help.");
	                   System.exit(1);
	                }
	            } catch (NumberFormatException e) {
	            	System.out.println(args[i] + 
	            			" looks like a malformed option.\n"
	                        + "Use option -h for help.");
	               System.exit(1);
	            }
            } // endfor args
		} // endif options

		// Now, getting down to business.
		HasVocals hasVocals = new HasVocals(System.out);
		hasVocals.newNeuralNetwork();
		if(dataFile != null) { 
			try {
				hasVocals.train(dataFile, 
							    audioDir, 
							    temp, 
							    recurse, 
							    n, 
							    minDeltaError, 
							    maxEpochs);
			} catch(FileNotFoundException e) {
				System.out.println(e.getMessage());
				System.exit(1);
			}
		} else {
			hasVocals.train(temp, recurse, n, minDeltaError, maxEpochs, maxThreads);
		}
		//hasVocals.saveNeuralNetwork(annOutputFile);
	}
	
	public static String getOptionParameter(String[] args, int i) {
		if(i + 1 >= args.length - MAIN_REQUIRED_ARGS) {
			System.out.println("Malformed args. Use option -h for help.");
			System.exit(1);
		}
		return args[i+1];
	}
	
	public static String getFiletype(File file) {
		String name = file.getName();
		int fileTypeIdx = name.lastIndexOf('.') + 1;
		String filetype = name.substring(fileTypeIdx);
		return filetype;
	}
	
	public static boolean isCSV(File file) {
		String filetype = getFiletype(file);
		return (filetype.equals("csv") || filetype.equals("CSV"));
	}
	
	public static boolean isWAV(File file) {
		String filetype = getFiletype(file);
		return (filetype.equals("wav") || filetype.equals("WAV"));
	}
	
	//--------------------------------------------------------------------------
	// INSTANCE STUFF
	//--------------------------------------------------------------------------
	
	private PrintStream mOut;
	private Mlp mNeuralNetwork;
	private List<File> mAudioFileList;
	private List<File> mDataFileList;
	private File mTemp;
	private HashMap<String, Double> mLabelsByFilename;
	private ArrayList<LabeledDataContainer> mTrainingContainers;
	
	public HasVocals(PrintStream out) {
		if(out != null)
			mOut = out;
	}
	
	public void newNeuralNetwork() {
		SoftMax softMax = new SoftMax(1);
		StandardLogistic logistic = new StandardLogistic(1);
		mNeuralNetwork = new Mlp();
		
		// Construct layer
		Mlp.Layer hidden1 = 
				new Mlp.Layer(30, logistic);
		Mlp.Layer hidden2 = 
				new Mlp.Layer(10, logistic);
		Mlp.Layer output = 
				new Mlp.Layer(1, softMax);
		
		// Link them up
		mNeuralNetwork.append(hidden1).append(hidden2).append(output);
	}
	
	public void setNeuralNetwork(Mlp n) {
		mNeuralNetwork = n;
	}
	

	public void train(File dataFile, 
					  File audioDir, 
					  File temp, 
					  boolean recurse, 
					  int n,
					  double minDeltaError,
					  int maxEpochs) 
		throws FileNotFoundException 
	{
		if(dataFile.isDirectory() || !isCSV(dataFile))
        	throw new IllegalArgumentException(
        					"Invalid csv datafile: " + dataFile.getPath());
        if(!audioDir.isDirectory())
        	throw new IllegalArgumentException(
        					"Not a directory: " + audioDir.getPath());
        mTemp = temp;
    	parseDataFile(dataFile);
        preprocessAudio(audioDir, recurse, n);
        clearTrainingState();
	}
	
	public void train(File trainingDir, 
					  boolean recurse, 
					  int n,
					  double minDeltaError,
					  int maxEpochs, 
					  int maxThreads) 
	{
		mTrainingContainers = new ArrayList<LabeledDataContainer>();
		String[] filetypes = new String[] {"mfc"};
		FileFilter filter =  new TrainingFileFilter(filetypes, null);
		List<File> fileList = new ArrayList<File>();
		processDir(trainingDir, fileList, filter, recurse);
		println("Found " + fileList.size() + " valid files");
		
		n = Math.min(n, fileList.size());
		println("Randomly selecting, at most, " 
						+ n + " with which to train.");
		Collections.shuffle(fileList);
		fileList.subList(n, fileList.size()).clear();
		for(File file : fileList) {
			mTrainingContainers.add(new LabeledDataContainer(file));
		}
		train(minDeltaError, maxEpochs, maxThreads);
	}
	
	private void train(double minDeltaError, int maxEpochs, int maxThreads) {
		int trainingSize = (int) (.75 * mTrainingContainers.size());
		int testingSize = mTrainingContainers.size() - trainingSize;
		
		println("Training size: " + trainingSize 
				+ ". Testing size: " + testingSize);
		
		LabeledDataContainer[] trainingSet = new LabeledDataContainer[trainingSize];
		mTrainingContainers
			.subList(0, trainingSize)
				.toArray(trainingSet);
		
		LabeledDataContainer[] testingSet = new LabeledDataContainer[testingSize];
		mTrainingContainers
			.subList(trainingSize, mTrainingContainers.size())
				.toArray(testingSet);
		
		MlpTrainer trainer = 
				new MlpTrainer(mNeuralNetwork, mOut);
		trainer.trainMlp(trainingSet, 
						     testingSet, 
						     minDeltaError, 
						     maxEpochs,
						     maxThreads);
	}
	
	private void parseDataFile(File file) throws FileNotFoundException {
		mLabelsByFilename = new HashMap<String, Double>();
		CSVReader reader = new CSVReader(new FileReader(file));
		try {
			

			String fileNameHeader = "Input.filename";
			String labelHeader = "Answer.label";
			String hasVocalsLabel = "has-vocals";
			String noVocalsLabel = "no-vocals";
			
			int filenameIdx = -1;
			int labelIdx = -1;
			
			// parse CSV header to determine locations of interesting values
			String[] headers = reader.readNext();
			
			for(int i=0; i < headers.length; i++) {
				if(headers[i].contains(fileNameHeader))
					filenameIdx = i;
				if(headers[i].contains(labelHeader))
					labelIdx = i;
			}
			
			if(filenameIdx < 0 || labelIdx < 0)
				throw new IllegalArgumentException(
						"The dataFile provided does not include " + 
						"the necessary headers.");
			
			String[] items = reader.readNext();
			while (items != null) {
				String filename = items[filenameIdx].trim();
				// Remove CSV quotation marks where necessary
				if(filename.startsWith("\""))
					filename = filename.substring(1, filename.length());
				if(filename.endsWith("\""))
					filename = filename.substring(1, filename.length());
				String label = items[labelIdx];
				Double dLabel = null;
				if(label.contains(hasVocalsLabel)) {
					dLabel = new Double(1D);
				} else if (label.contains(noVocalsLabel)){
					dLabel = new Double(0D);
				}
				if(dLabel != null) {
					println("putting " + filename + "in as " + dLabel);
					mLabelsByFilename.put(filename, dLabel);
				} else {
					println("failed to get label for " + filename);
				}
				 items = reader.readNext();
			}

		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {reader.close();} catch (IOException e) {}
		}
	}
	
	private void preprocessAudio(File root, boolean recurse, int n) {
		long start = System.currentTimeMillis();
		println("Preprocessing audio...");
		mTrainingContainers = new ArrayList<LabeledDataContainer>();
		List<File> audioFileList = selectFiles(root, recurse, n);
		for(int i=0; i < audioFileList.size();i++) {
			File file = audioFileList.get(i);
			// Get filename without filetype
			String name = file.getName();
			print("Processing (" + i + "/" + audioFileList.size() + ") " 
					   + name + "\r");
			
			int fileTypeIdx = name.lastIndexOf('.');
			name = name.substring(0, fileTypeIdx);
			// Get this file's label from dataFile hashmap
			Double doubleLabel = mLabelsByFilename.get(name);
			if(doubleLabel == null) {
				println("WARNING: Label not found for: " + name);
				continue;
			}
			double label = doubleLabel.doubleValue();
			
			SpeechDataContainer prepContainer = null;
			try {
			
				prepContainer = new SpeechDataContainer(file, label);
				List<LabeledData> dataList = new ArrayList<LabeledData>();
				
				prepContainer.open();
				while(prepContainer.hasNext()) {
					dataList.add(prepContainer.next());
				}
				
				
				String outputPath = mTemp.getAbsolutePath() 
						+ File.separator
						+ name
						+ ".mfc";
				File mfcFile = new File(outputPath);
	
				LabeledData.writeToFile(dataList, mfcFile, false);
				// TODO create container instead
				mTrainingContainers.add(new LabeledDataContainer(mfcFile));
				
			} catch (IOException e2) {
				e2.printStackTrace();
			} catch (IDataContainer.DataUnavailableException e2) {
				e2.printStackTrace();
			} finally {
				if(prepContainer != null)
					try {prepContainer.close();} catch (Exception e) {}
			}
		}
		long elapsed = System.currentTimeMillis() - start;
		println("Finished preprocessing audio. " + elapsed + " ms.");
	}
	
	private List<File> selectFiles(File root, boolean recurse, int n) {
		List<File> list = new ArrayList<File>();
		String[] filetypes = new String[] {"wav"};
		FileFilter filter = 
				new TrainingFileFilter(filetypes, mLabelsByFilename);
		
		processDir(root, list, filter, recurse);
		println("Found " + list.size() + " valid files");
		
		n = Math.min(n, list.size());
		println("Randomly selecting " + n + " with which to train.");
		Collections.shuffle(list);
		list.subList(n, list.size()).clear();
		
		return list;
	}
	
	private void clearTrainingState() {
		mTemp = null;
		mLabelsByFilename = null;
		mTrainingContainers = null;
		// TODO delete temporary files
	}
	
	public void saveNeuralNetwork(File file) {
	}
	
	public void loadNeuralNetwork(File file) {
	}
	
	public boolean hasVocals(File file) {
		if(!isWAV(file))
			throw new IllegalArgumentException(
					"Not a wav file: " + file.getPath());
		
		if(mNeuralNetwork == null)
			throw new IllegalStateException(
					"You must load or train a neural network before"
					+ " testing a file for vocals");
		
		return false;
	}
	
	private void processDir(File node, 
							List<File> list,
            				FileFilter filter, 
            				boolean recurse, 
            				int depth)
    {
		if(node.isDirectory()) {
            // Only recurse if we're supposed to
            if(recurse || depth ==0) {
                File[] nodeFiles = node.listFiles(filter);
                for(File file : nodeFiles){
                    processDir(file, list, filter, recurse, depth+1);
                }
            }
        } else if(filter.accept(node)){
            list.add(node);
        }
    }
	
	private void processDir(File node, 
							List<File> list,
            				FileFilter filter,
            				boolean recurse) 
    {
        println("Searching " + node.getAbsolutePath());
        processDir(node, list, filter, recurse, 0);
    }
	
	public void print(String str) {
		if(mOut != null)
			mOut.print(str);
	}
	
	public void println(String str) {
		if(mOut != null)
			mOut.println(str);
	}
	
	/**
     * An AudioFileFilter... omg! so punny.
     * @author Michael DuBois
     *
     */
    public static class TrainingFileFilter implements FileFilter {

        List<String> mFiletypes;
		private HashMap<String, Double> mLabels;
        
        public TrainingFileFilter(String[] args, 
        						  HashMap<String,Double> labels) 
        {
            mFiletypes = new ArrayList<String>(Arrays.asList(args));
            mLabels = labels;
        }
        
        @Override
        public boolean accept(File file) {
            String name = file.getName();
            int typeIdx = name.lastIndexOf('.', name.length() - 1) + 1;
            String type = name.substring(typeIdx);
            name = name.substring(0, typeIdx - 1);
            boolean labeled = true;
            if(mLabels != null)
            	labeled = (mLabels.get(name) != null);
            boolean isAudio = false;
            if(mFiletypes.contains(type))
                isAudio = true;
            return file.isDirectory() || (isAudio && labeled);
        }
        
    }
}
