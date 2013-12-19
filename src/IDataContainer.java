
public interface IDataContainer {
	public void open() throws DataUnavailableException;
	public void close() throws DataUnavailableException;
	public boolean hasNext();
	public LabeledData next() throws DataUnavailableException;
	
	public static class DataUnavailableException extends Exception {
		DataUnavailableException(String msg) {
			super(msg);
		}
		private static final long serialVersionUID = 8392846021695850077L;
	}
}
