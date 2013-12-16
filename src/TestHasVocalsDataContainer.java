import java.io.File;
import java.io.IOException;


public class TestHasVocalsDataContainer {

	public static void main(String[] args) throws Exception {
		if(args.length <= 0) {
			System.out.println("Please specify a wav filepath.");
			System.exit(1);
		}
		File file = new File(args[0]);
		WavFile wavFile = WavFile.openWavFile(file);
		wavFile.display();
		
		int num = Integer.MAX_VALUE;
		if(args.length > 1)
			num = Integer.parseInt( args[1] );
		
		
		HasVocalsDataContainer container = 
				new HasVocalsDataContainer(wavFile, 1);
		System.out.println("WindowSize: " + container.getWindowSize());
		System.out.println("ShiftSize: " + container.getShiftSize());
		long start = System.currentTimeMillis();
		container.open();
		int windows = 0;
		while(container.hasNext() && windows < num) {
			LabeledData datum = container.next();
			//System.out.println( datum.toString() );
			
			windows++;
		}
		long elapsed = System.currentTimeMillis() - start;
		System.out.println("processed " + windows + " windows");
		System.out.println("elapsed: " + elapsed + "ms");
	}
}
