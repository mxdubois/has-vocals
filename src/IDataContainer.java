/**
 * An interface for acquiring data that may require some preprocessing or
 * file I/O.
 * @author Michael DuBois
 *
 */
public interface IDataContainer {
	
	public void open() throws DataUnavailableException;
	public void close() throws DataUnavailableException;
	public boolean hasNext();
	public LabeledData next() throws DataUnavailableException;
	
	/**
	 * An exception that IDataContainers can throw when something goes wrong.
	 * @author Michael DuBois
	 *
	 */
	public static class DataUnavailableException extends Exception {
		DataUnavailableException(String msg) {
			super(msg);
		}
		private static final long serialVersionUID = 8392846021695850077L;
	}
}
