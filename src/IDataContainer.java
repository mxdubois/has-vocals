
public interface IDataContainer {
	public void open() throws Exception;
	public void close() throws Exception;
	public boolean hasNext();
	public LabeledData next() throws Exception;
}
