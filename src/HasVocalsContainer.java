import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class HasVocalsContainer implements IDataContainer {

	private File mFile;
	private LabeledData[] mDataList;
	private int mIdx;

	public HasVocalsContainer(File file) {
		mFile = file;
	}

	@Override
	public void open() throws Exception {
		mDataList = LabeledData.readFromFile(mFile);
		mIdx = 0;
	}

	@Override
	public void close() throws Exception {
		mDataList = null;
	}

	@Override
	public boolean hasNext() {
		return mIdx < mDataList.length - 1;
	}

	@Override
	public LabeledData next() throws Exception {
		LabeledData next = mDataList[mIdx];
		mIdx++;
		return next;
	}
}
