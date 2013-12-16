import java.io.File;


public class HasVocals {

	public static final String TAG = "HasVocals";
	
	public static void main(String[] args) throws IllegalArgumentException {
		int numRequiredArgs = 2;
		
		if(args.length < numRequiredArgs) {
            throw new IllegalArgumentException("You didn't supply both " 
            								+ "a srcdir and output dir.");
        }
		
		//Required args
        File dataFile = new File(args[args.length - 2]);
        File audioDir = new File(args[args.length - 1]);

		// Option defaults
		boolean recurse = false;
		
		// Things that could one day be options
		String[] filetypes = new String[] {"wav"};
		
		if(args.length > numRequiredArgs) {
            // Get options
            for(int i=0; i < args.length - numRequiredArgs; i++) {
                if(args[i].startsWith("-") && args[i].length() == 2) {
                    // It is a valid option
                    char option = args[i].charAt(1);
                    switch(option) {
                    case 'h':
                        // Display help
                        System.out.println(
                                "Usage: " + TAG 
                                + "[options] dataFile audioDir \n"
                                + "-r|-R search srcdir recursively"
                                + "-h display this help message"
                                );
                        System.exit(0);
                    case 'r' :
                    case 'R' :
                        // Recurse on directories
                        recurse = true;
                        break;
                    default :
                    	throw new IllegalArgumentException(
                                					"Invalid flag " + args[i]);
                    }
                } else {
                    throw new IllegalArgumentException(
                            args[i] + " looks like a malformed option.\n"
                            + "Use option -h for help.");
                }
            } // endfor
		} // endif options
		
		HasVocals hasVocals = new HasVocals(dataFile, audioDir, recurse);
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
	
	//--------------------------------------------------------------------------
	// INSTANCE STUFF
	//--------------------------------------------------------------------------
	
	public HasVocals(File dataFile, File audioDir, boolean dirRecurse) {
        if(dataFile.isDirectory() || !isCSV(dataFile))
        	throw new IllegalArgumentException(
        					"Invalid csv datafile: " + dataFile.getPath());
        if(!audioDir.isDirectory())
        	throw new IllegalArgumentException(
        					"Not a directory: " + audioDir.getPath());
        
        // Initialize neural net
	}
	
}
