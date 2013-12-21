import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * A DataContainer that reads LabeledData from a file.
 * @author Michael DuBois
 *
 */
public class LabeledDataContainer implements IDataContainer {

	private File mFile;
	private LabeledData[] mDataList;
	private int mIdx;

	/**
	 * Constructs a LabeledDataContainer
	 * @param file
	 */
	public LabeledDataContainer(File file) {
		mFile = file;
	}

	@Override
	public void open() throws DataUnavailableException {
		mDataList = LabeledData.readFromFile(mFile);
		mIdx = 0;
	}

	@Override
	public void close() throws DataUnavailableException {
		mDataList = null;
	}

	@Override
	public boolean hasNext() {
		return mIdx < mDataList.length - 1;
	}

	@Override
	public LabeledData next() throws DataUnavailableException {
		LabeledData next = mDataList[mIdx];
		mIdx++;
		return next;
	}
}
